"""Convert timm DINOv3 ViT-S / ViT-S+ weights (HF 'timm/' hub, no gated access) to the official DINOv3 key format
used by DEIMv2 (engine/backbone/dinov3), and verify the conversion numerically.

python tools/steel/convert_dinov3_timm.py {dinov3_vits16|dinov3_vits16plus} out.pth [--verify]
(scripts/download_weights.sh does this automatically for DEIMv2-L / X)
"""

import sys
from pathlib import Path

import torch

TIMM_NAMES = {
    "dinov3_vits16": "vit_small_patch16_dinov3_qkvb.lvd1689m",
    "dinov3_vits16plus": "vit_small_plus_patch16_dinov3_qkvb.lvd1689m",
}


def convert(timm_sd, ref_sd):
    out = {}
    for k in [k for k in timm_sd if k.endswith("attn.q_bias")]:  # timm keeps q/v biases apart; official: qkv.bias
        pre = k[: -len("q_bias")]
        q, v = timm_sd[pre + "q_bias"], timm_sd[pre + "v_bias"]
        out[pre + "qkv.bias"] = torch.cat([q, torch.zeros_like(q), v])
    for k, v in timm_sd.items():
        if k.endswith(("attn.q_bias", "attn.v_bias")):
            continue
        k2 = (k.replace("reg_token", "storage_tokens").replace(".gamma_1", ".ls1.gamma").replace(".gamma_2", ".ls2.gamma")
               .replace(".mlp.fc1_g.", ".mlp.w1.").replace(".mlp.fc1_x.", ".mlp.w2.").replace(".mlp.fc2.", ".mlp.w3."
                        if any(".mlp.w1." in r for r in ref_sd) else ".mlp.fc2."))
        out[k2] = v
    for k, v in ref_sd.items():
        if k.endswith("attn.qkv.bias_mask"):  # official DINOv3 masks the key bias: [q=1, k=0, v=1]
            c = v.shape[0] // 3
            m = torch.ones_like(v)
            m[c:2 * c] = 0
            out[k] = m
        elif k not in out:  # mask_token, rope periods: keep the model's own values
            out[k] = v
    missing = set(ref_sd) - set(out)
    extra = set(out) - set(ref_sd)
    assert not missing and not extra, (missing, extra)
    for k in ref_sd:
        assert out[k].shape == ref_sd[k].shape, (k, out[k].shape, ref_sd[k].shape)
    return out


if __name__ == "__main__":
    import timm

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from engine.backbone.dinov3.vision_transformer import DinoVisionTransformer

    name, out_path = sys.argv[1], sys.argv[2]
    tm = timm.create_model(TIMM_NAMES[name], pretrained=True).eval()
    dm = DinoVisionTransformer(name=name).eval()
    sd = convert(tm.state_dict(), dm.state_dict())
    dm.load_state_dict(sd)
    torch.save(sd, out_path)
    print(f"saved {out_path}: {len(sd)} tensors")
    if "--verify" in sys.argv:
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            a = tm.forward_features(x)[:, tm.num_prefix_tokens:]  # patch tokens after final norm
            b = dm.forward_features(x)[0]["x_norm_patchtokens"] if isinstance(dm.forward_features(x), list) \
                else dm.forward_features(x)["x_norm_patchtokens"]
        diff = (a - b).abs().max().item()
        print(f"verify: max |timm - deimv2| over patch tokens = {diff:.2e} (cosine "
              f"{torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item():.6f})")
