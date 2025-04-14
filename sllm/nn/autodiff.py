"""
This module implements custom autograd functions for performing matrix multiplications,
including LoRA-adapted operations for efficient fine-tuning. It leverages lazy weight loading,
bundled scaled matrix multiplications, and gradient clipping to manage memory and computational
efficiency.

Functions:
    clip_grad(grad): Clips a gradient tensor based on MAX_GRAD_NORM.
    
Classes:
    MatmulFunction: Custom autograd function for computing scaled matrix multiplications.
    BundledMatmulFunction: Custom autograd function that computes multiple scaled matrix multiplications together.
    LoraFunction: Custom autograd function to perform a LoRA-adapted linear operation.
    LoraQKVLinearFunction: Custom autograd function for query, key, and value projections with LoRA adaptation.
"""

import os

import torch
import torch.nn.functional as F

from sllm.common import GRADIENT_DIR, MAX_GRAD_NORM
from sllm.ops import bundled_scaled_matmul
from sllm.utils import load_tensor_from_storage, save_tensor_to_storage


def clip_grad(grad):
    """
    Clip a gradient tensor to prevent exploding gradients.

    If the norm of `grad` exceeds MAX_GRAD_NORM, scale it down accordingly.

    Args:
        grad (Tensor or None): The gradient tensor to be clipped.

    Returns:
        Tensor or None: The clipped gradient tensor, or None if input is None.
    """
    if grad is None:
        return None
    norm = grad.norm()
    if norm > MAX_GRAD_NORM:
        grad = grad * (MAX_GRAD_NORM / (norm + 1e-6))
    return grad


class MatmulFunction(torch.autograd.Function):
    """
    Custom autograd function for performing a scaled matrix multiplication.

    Forward:
        Computes Y = bundled_scaled_matmul([(A, B, scale)])[0], which is equivalent to
        Y = (A @ B) * scale.

    Backward Derivation:
        Given Y = scale * (A @ B), standard matrix calculus yields:
            - dY/dA = scale * grad_output @ B^T 
            - dY/dB = scale * A^T @ grad_output
        Since the scale is a constant and non-differentiable here, its derivative is not returned
        (i.e. None). In this implementation, the bundled_scaled_matmul function is used to compute:
            grad_A = grad_output @ B^T   and   grad_B = A^T @ grad_output,
        assuming that the scale factor is applied in the forward pass and treated as a constant.
    """

    @staticmethod
    def forward(ctx, A, B, scale):
        """
        Forward pass for the scaled matrix multiplication.

        Args:
            A (Tensor): Left-hand side matrix.
            B (Tensor): Right-hand side matrix.
            scale (float): Scaling factor applied after matrix multiplication.

        Returns:
            Tensor: Result of the scaled matrix multiplication.
        """
        ctx.save_for_backward(A, B)
        return bundled_scaled_matmul([(A, B, scale)])[0]

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the scaled matrix multiplication.

        Derivation:
            Let Y = scale * (A @ B). Then, by the chain rule:
                dL/dA = dL/dY @ dY/dA = grad_output @ (B^T) * scale,
                dL/dB = (A^T) @ grad_output * scale.
            Here, the implementation uses bundled_scaled_matmul with a scaling factor of 1.0,
            effectively treating the scale as a constant whose derivative is omitted.
        
        Args:
            grad_output (Tensor): Gradient tensor propagated from subsequent layers.

        Returns:
            Tuple[Tensor, Tensor, None]: Gradients with respect to A, B, and None for scale.
        """
        A, B = ctx.saved_tensors
        bundles = [
            (grad_output, B.transpose(-2, -1), 1.0),  # represents grad_output @ B^T
            (A.transpose(-2, -1), grad_output, 1.0),  # represents A^T @ grad_output
        ]
        grad_A, grad_B = bundled_scaled_matmul(bundles)
        grad_A = clip_grad(grad_A)
        grad_B = clip_grad(grad_B)
        return grad_A, grad_B, None


class BundledMatmulFunction(torch.autograd.Function):
    """
    Custom autograd function for performing multiple scaled matrix multiplications in batch.

    Forward:
        Accepts a list of bundles where each bundle is a tuple (M, N, scale) and returns a list
        of results computed by bundled_scaled_matmul.

    Backward Derivation:
        For each bundle where the forward operation is Y_i = M_i @ N_i * scale_i, the gradient
        derivation uses the standard identities:
            dL/dM_i = grad_output @ (N_i^T)
            dL/dN_i = (M_i^T) @ grad_output
        Here, a new bundle is constructed for each stored tuple in the forward pass with a fixed scale (1.0)
        and processed in batch.
    """

    @staticmethod
    def forward(ctx, bundles):
        """
        Forward pass for the bundled scaled matrix multiplications.

        Args:
            bundles (List[Tuple[Tensor, Tensor, float]]): A list of tuples, each containing two tensors
                and a scaling factor.

        Returns:
            List[Tensor]: A list of tensors resulting from the matrix multiplications.
        """
        ctx.save_for_backward(*bundles)
        return bundled_scaled_matmul(bundles)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the bundled matrix multiplications.

        Derivation:
            For each bundle (M, N, scale) corresponding to Y = M @ N * scale, the gradient with respect
            to M is grad_output @ (N^T) and with respect to N is (M^T) @ grad_output. The backward pass
            constructs a new list of bundles where for each saved bundle it forms:
                (grad_output, N^T, 1.0)
            and computes the corresponding gradients in one batched call.
        
        Args:
            grad_output (Tensor): Gradient tensor propagated from subsequent layers.

        Returns:
            Tuple[List[Tensor]]: Gradients corresponding to the input bundles.
        """
        bundles = ctx.saved_tensors
        triple_list = [
            (grad_output, bundle[1].transpose(-2, -1), 1.0) for bundle in bundles
        ]
        grad_bundles = bundled_scaled_matmul(triple_list)
        return grad_bundles


class LoraFunction(torch.autograd.Function):
    """
    Custom autograd function for LoRA-adapted linear operations.

    Forward:
        Computes the effective weight as W_eff = W + (A @ B * scale) and then computes:
            Y = bundled_scaled_matmul([(x, W_eff^T, 1.0)])[0]
        Optionally adds a bias.

    Backward Derivation:
        Let W_eff = W + (A @ B * scale) and Y = x @ (W_eff)^T.
        Using the chain rule:
            dL/dx = grad_output @ W_eff
            dL/dW_eff = x^T @ grad_output
        Then, the derivatives with respect to A and B are computed via:
            dW_eff/dA = B * scale,    dW_eff/dB = A * scale.
        The backward pass first computes an intermediate gradient E from the matrix multiplication,
        then derives:
            grad_A = E @ (B^T) * scale
            grad_B = (A^T) @ E * scale.
        The pre-trained weight W is assumed fixed, so its gradient is not computed.
    """

    @staticmethod
    def forward(ctx, x, A, B, W_path, scale, bias=None):
        """
        Forward pass for the LoRA function.

        Args:
            x (Tensor): Input tensor.
            A (Tensor): LoRA parameter A.
            B (Tensor): LoRA parameter B.
            W_path (str): File path to the pre-trained weight tensor.
            scale (float): Scaling factor for the LoRA update.
            bias (Tensor, optional): Bias tensor to add to the output.

        Returns:
            Tensor: Output tensor after applying the effective weight and bias.
        """
        ctx.save_for_backward(A, B)
        ctx.scale = scale
        ctx.x_path = os.path.join(GRADIENT_DIR, W_path.split("/")[-1] + ".x.bin")
        save_tensor_to_storage(ctx.x_path, x)

        W = load_tensor_from_storage(
            weight_path=W_path,
            shape=(A.shape[0], B.shape[1]),
            dtype=A.dtype,
            to_ram=False,
        )

        effective_W = W + (A @ B * scale)
        Wx = bundled_scaled_matmul([(x, effective_W.transpose(-2, -1), 1.0)])[0]

        ctx.effecttive_weight = effective_W

        if bias is not None:
            Wx = Wx + bias
        return Wx

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the LoRA function.

        Derivation:
            Given:
                Y = x @ (W_eff)^T,   with   W_eff = W + (A @ B) * scale.
            Then:
                dL/dx = grad_output @ W_eff.
                dL/dW_eff = x^T @ grad_output.
            Since W is fixed, we only differentiate through the LoRA update.
            The gradient of W_eff with respect to A is: dW_eff/dA = B * scale,
            and with respect to B is: dW_eff/dB = A * scale.
            Therefore, an intermediate gradient E = dL/dW_eff is computed and then:
                grad_A = E @ (B^T) * scale,
                grad_B = (A^T) @ E * scale.
        
        Args:
            grad_output (Tensor): Gradient tensor from subsequent operations.

        Returns:
            Tuple: Gradients with respect to x, A, B, None (for weight), None (for scale), and None (for bias).
        """
        x = load_tensor_from_storage(
            ctx.x_path, shape=grad_output.shape, dtype=grad_output.dtype, to_ram=False
        )
        A, B = ctx.saved_tensors
        scale = ctx.scale

        effective_W = ctx.effecttive_weight

        bundles = [
            (grad_output, effective_W, 1.0),         # dL/dx = grad_output @ W_eff
            (x.transpose(-2, -1), grad_output, 1.0),   # dL/dW_eff = x^T @ grad_output
        ]
        grad_x, E = bundled_scaled_matmul(bundles)

        bundles = [
            (E, B.transpose(-2, -1), scale),          # grad_A = E @ B^T * scale
            (A.transpose(-2, -1), E, scale)            # grad_B = A^T @ E * scale
        ]
        grad_A, grad_B = bundled_scaled_matmul(bundles)

        grad_w = None
        grad_scale = None
        grad_b = None

        return grad_x, grad_A, grad_B, grad_w, grad_scale, grad_b


class LoraQKVLinearFunction(torch.autograd.Function):
    """
    Custom autograd function for LoRA-adapted query/key/value projections.

    Forward:
        For each projection (Q, K, V), the effective weight is computed as:
            effective = weight + (LoRA_A @ LoRA_B * scaling)
        and the projections are computed as:
            Projection = bundled_scaled_matmul([(x, effective, 1.0)])
        Biases are added if provided.

    Backward Derivation:
        Let Q = x @ (q_effective)^T + bias, with q_effective = q_weight + (q_proj_lora_A @ q_proj_lora_B * scaling).
        By the chain rule:
            dL/dx = sum_{proj in {Q, K, V}} [ (x^T)' from that projection ],
        where for each projection the gradients with respect to the effective weights are obtained by:
            dL/d(effective) = x^T @ grad_projection.
        Then, the gradients with respect to the low-rank parameters (LoRA_A and LoRA_B) are computed
        using:
            grad_LoRA_A = dL/d(effective) @ (LoRA_B)^T * scaling,
            grad_LoRA_B = (LoRA_A)^T @ dL/d(effective) * scaling.
        The input x gradient is computed as the sum of contributions from Q, K, and V pathways.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        q_proj_weight_path,
        k_proj_weight_path,
        v_proj_weight_path,
        q_proj_bias,
        k_proj_bias,
        v_proj_bias,
        q_proj_lora_A,
        q_proj_lora_B,
        k_proj_lora_A,
        k_proj_lora_B,
        v_proj_lora_A,
        v_proj_lora_B,
        scaling,
    ):
        """
        Forward pass for the LoRA QKV function.

        Args:
            x (Tensor): Input tensor.
            q_proj_weight_path (str): File path for the query projection weight.
            k_proj_weight_path (str): File path for the key projection weight.
            v_proj_weight_path (str): File path for the value projection weight.
            q_proj_bias (Tensor or None): Bias tensor for the query projection.
            k_proj_bias (Tensor or None): Bias tensor for the key projection.
            v_proj_bias (Tensor or None): Bias tensor for the value projection.
            q_proj_lora_A (Tensor): LoRA parameter A for query projection.
            q_proj_lora_B (Tensor): LoRA parameter B for query projection.
            k_proj_lora_A (Tensor): LoRA parameter A for key projection.
            k_proj_lora_B (Tensor): LoRA parameter B for key projection.
            v_proj_lora_A (Tensor): LoRA parameter A for value projection.
            v_proj_lora_B (Tensor): LoRA parameter B for value projection.
            scaling (float): Scaling factor for the LoRA update.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: The projected query, key, and value tensors.
        """
        ctx.save_for_backward(
            q_proj_lora_A,
            q_proj_lora_B,
            k_proj_lora_A,
            k_proj_lora_B,
            v_proj_lora_A,
            v_proj_lora_B,
        )

        ctx.scaling = scaling
        ctx.q_bias_flag = q_proj_bias is not None
        ctx.k_bias_flag = k_proj_bias is not None
        ctx.v_bias_flag = v_proj_bias is not None

        # Save input x shape for later and store x on disk.
        ctx.x_shape = x.shape
        ctx.x_path = os.path.join(
            GRADIENT_DIR, q_proj_weight_path.split("/")[-1] + ".x.bin"
        )
        save_tensor_to_storage(ctx.x_path, x)

        q_shape = (q_proj_lora_A.shape[0], x.shape[-1])
        kv_shape = (k_proj_lora_B.shape[-1], x.shape[-1])
        ctx.q_shape = q_shape
        ctx.kv_shape = kv_shape

        q_proj_weight = load_tensor_from_storage(
            weight_path=q_proj_weight_path,
            shape=q_shape,
            dtype=q_proj_lora_A.dtype,
            to_ram=False,
        ).transpose(-2, -1)
        k_proj_weight = load_tensor_from_storage(
            weight_path=k_proj_weight_path,
            shape=kv_shape,
            dtype=k_proj_lora_A.dtype,
            to_ram=False,
        ).transpose(-2, -1)
        v_proj_weight = load_tensor_from_storage(
            weight_path=v_proj_weight_path,
            shape=kv_shape,
            dtype=v_proj_lora_A.dtype,
            to_ram=False,
        ).transpose(-2, -1)

        # Compute effective weights with LoRA update.
        q_effective = q_proj_weight + (q_proj_lora_A @ q_proj_lora_B * scaling)
        k_effective = k_proj_weight + (k_proj_lora_A @ k_proj_lora_B * scaling)
        v_effective = v_proj_weight + (v_proj_lora_A @ v_proj_lora_B * scaling)

        ctx.q_effective = q_effective
        ctx.k_effective = k_effective
        ctx.v_effective = v_effective

        bundles = [
            (x, q_effective, 1.0),
            (x, k_effective, 1.0),
            (x, v_effective, 1.0),
        ]

        Q, K, V = bundled_scaled_matmul(bundles)

        if q_proj_bias is not None:
            Q = Q + q_proj_bias
        if k_proj_bias is not None:
            K = K + k_proj_bias
        if v_proj_bias is not None:
            V = V + v_proj_bias

        return Q, K, V

    @staticmethod
    def backward(ctx, grad_Q, grad_K, grad_V):
        """
        Backward pass for the LoRA QKV function.

        Derivation:
            For each projection (Q, K, V), let:
                effective = weight + (LoRA_A @ LoRA_B * scaling)
            and the forward operation is:
                Projection = x @ (effective)^T (+ bias).
            The gradients are computed as follows:
                1. Compute dL/d(effective) for each projection by:
                    dL/d(effective) = x^T @ grad_projection
                2. The gradient with respect to x is obtained by summing over contributions:
                    grad_x = grad_xQ + grad_xK + grad_xV
                3. Using the chain rule and the linearity of the LoRA update:
                    grad_LoRA_A = dL/d(effective) @ (LoRA_B)^T * scaling,
                    grad_LoRA_B = (LoRA_A)^T @ dL/d(effective) * scaling.
            Here, the bundled_scaled_matmul function is used to compute both the gradients for x
            (from each of Q, K, V) and the gradients for the effective weights, which are then propagated
            to the low-rank LoRA parameters.
        
        Args:
            grad_Q (Tensor): Gradient with respect to the query output.
            grad_K (Tensor): Gradient with respect to the key output.
            grad_V (Tensor): Gradient with respect to the value output.

        Returns:
            Tuple: Gradients for each input parameter in the same order as in the forward pass.
        """
        (
            q_proj_lora_A,
            q_proj_lora_B,
            k_proj_lora_A,
            k_proj_lora_B,
            v_proj_lora_A,
            v_proj_lora_B,
        ) = ctx.saved_tensors
        scale = ctx.scaling
        x = load_tensor_from_storage(
            ctx.x_path, shape=ctx.x_shape, dtype=grad_Q.dtype, to_ram=False
        )

        q_effective = ctx.q_effective
        k_effective = ctx.k_effective
        v_effective = ctx.v_effective

        bundles = [
            (grad_Q, q_effective.transpose(-2, -1), 1.0),
            (grad_K, k_effective.transpose(-2, -1), 1.0),
            (grad_V, v_effective.transpose(-2, -1), 1.0),
            (x.transpose(-2, -1), grad_Q, 1.0),
            (x.transpose(-2, -1), grad_K, 1.0),
            (x.transpose(-2, -1), grad_V, 1.0),
        ]
        (
            grad_xQ,
            grad_xK,
            grad_xV,
            grad_effective_q,
            grad_effective_k,
            grad_effective_v,
        ) = bundled_scaled_matmul(bundles)

        grad_x = grad_xQ + grad_xK + grad_xV

        grad_q_bias = grad_Q.sum(dim=0) if ctx.q_bias_flag else None
        grad_k_bias = grad_K.sum(dim=0) if ctx.k_bias_flag else None
        grad_v_bias = grad_V.sum(dim=0) if ctx.v_bias_flag else None

        bundles = [
            (grad_effective_q, q_proj_lora_B.transpose(-2, -1), scale),
            (q_proj_lora_A.transpose(-2, -1), grad_effective_q, scale),
            (grad_effective_k, k_proj_lora_B.transpose(-2, -1), scale),
            (k_proj_lora_A.transpose(-2, -1), grad_effective_k, scale),
            (grad_effective_v, v_proj_lora_B.transpose(-2, -1), scale),
            (v_proj_lora_A.transpose(-2, -1), grad_effective_v, scale),
        ]
        grad_q_A, grad_q_B, grad_k_A, grad_k_B, grad_v_A, grad_v_B = bundled_scaled_matmul(
            bundles
        )

        grad_x = clip_grad(grad_x)
        grad_q_bias = clip_grad(grad_q_bias)
        grad_k_bias = clip_grad(grad_k_bias)
        grad_v_bias = clip_grad(grad_v_bias)
        grad_q_A = clip_grad(grad_q_A)
        grad_q_B = clip_grad(grad_q_B)
        grad_k_A = clip_grad(grad_k_A)
        grad_k_B = clip_grad(grad_k_B)
        grad_v_A = clip_grad(grad_v_A)
        grad_v_B = clip_grad(grad_v_B)
        grad_effective_q = clip_grad(grad_effective_q)
        grad_effective_k = clip_grad(grad_effective_k)
        grad_effective_v = clip_grad(grad_effective_v)
        grad_xQ = clip_grad(grad_xQ)
        grad_xK = clip_grad(grad_xK)
        grad_xV = clip_grad(grad_xV)
        grad_Q = clip_grad(grad_Q)
        grad_K = clip_grad(grad_K)
        grad_V = clip_grad(grad_V)

        grad_q_weight_path = None
        grad_k_weight_path = None
        grad_v_weight_path = None
        grad_scale = None

        return (
            grad_x,
            grad_q_weight_path,
            grad_k_weight_path,
            grad_v_weight_path,
            grad_q_bias,
            grad_k_bias,
            grad_v_bias,
            grad_q_A,
            grad_q_B,
            grad_k_A,
            grad_k_B,
            grad_v_A,
            grad_v_B,
            grad_scale,
        )
