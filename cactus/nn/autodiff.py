import os

import torch
import torch.nn.functional as F

from cactus.common import GRADIENT_DIR, MAX_GRAD_NORM
from cactus.nn.ops import cactus_matmul
from cactus.utils import load_tensor_from_storage, save_tensor_to_storage


def clip_grad(grad):
    if grad is None:
        return None
    norm = grad.norm()
    if norm > MAX_GRAD_NORM:
        grad = grad * (MAX_GRAD_NORM / (norm + 1e-6))
    return grad


class CactusMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, scale):
        ctx.save_for_backward(A, B)
        return cactus_matmul([(A, B, scale)])[0]

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        bundles = [
            (grad_output, B.transpose(-2, -1), 1.0),
            (A.transpose(-2, -1), grad_output, 1.0),
        ]
        grad_A, grad_B = cactus_matmul(bundles)
        grad_A = clip_grad(grad_A)
        grad_B = clip_grad(grad_B)
        return grad_A, grad_B, None


class CactusBundledMatmulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bundles):
        ctx.save_for_backward(*bundles)
        return cactus_matmul(bundles)

    @staticmethod
    def backward(ctx, grad_output):
        bundles = ctx.saved_tensors
        triple_list = [
            (grad_output, bundle[1].transpose(-2, -1), 1.0) for bundle in bundles
        ]
        grad_bundles = cactus_matmul(triple_list)
        return grad_bundles


class CactusLoraFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, A, B, W_path, scale, bias=None):
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
        Wx = cactus_matmul([(x, effective_W.transpose(-2, -1), 1.0)])[0]

        ctx.effecttive_weight = effective_W

        if bias is not None:
            Wx = Wx + bias
        return Wx

    @staticmethod
    def backward(ctx, grad_output):
        x = load_tensor_from_storage(
            ctx.x_path, shape=grad_output.shape, dtype=grad_output.dtype, to_ram=False
        )
        A, B = ctx.saved_tensors
        scale = ctx.scale

        effective_W = ctx.effecttive_weight

        bundles = [
            (grad_output, effective_W, 1.0),
            (x.transpose(-2, -1), grad_output, 1.0),
        ]
        grad_x, E = cactus_matmul(bundles)

        bundles = [(E, B.transpose(-2, -1), scale), (A.transpose(-2, -1), E, scale)]
        grad_A, grad_B = cactus_matmul(bundles)

        grad_w = None
        grad_scale = None
        grad_b = None

        return grad_x, grad_A, grad_B, grad_w, grad_scale, grad_b


class CactusLoraQKVLinearFunction(torch.autograd.Function):
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

        Q, K, V = cactus_matmul(bundles)

        if q_proj_bias is not None:
            Q = Q + q_proj_bias
        if k_proj_bias is not None:
            K = K + k_proj_bias
        if v_proj_bias is not None:
            V = V + v_proj_bias

        return Q, K, V

    @staticmethod
    def backward(ctx, grad_Q, grad_K, grad_V):
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
        ) = cactus_matmul(bundles)

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
        grad_q_A, grad_q_B, grad_k_A, grad_k_B, grad_v_A, grad_v_B = cactus_matmul(
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
