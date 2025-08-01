import os
import json
import csv
from pathlib import Path

# 定义根路径和数据集
ROOT_DIR = "/root/autodl-tmp/hai/VGGT-4D-baseline/third_party/4DGaussians/output"
METHODS = ["vggt", "monst3r"]
DATASETS = ["nerfie", "nvidia"]
SCENES = {
    "nerfie": ["chess4", "dvd", "hand8", "laptop8", "tomato-mark8"],
    "nvidia": ["Balloon1", "Balloon2", "Jumping", "Playground", "Skating", "Truck", "Umbrella"]
}

# 指标字段
METRICS = ["SSIM", "PSNR", "LPIPS-vgg"]

# 创建输出目录
os.makedirs("scripts", exist_ok=True)
csv_path = "scripts/result.csv"

# 存储结果
results = {
    "nerfie": {},
    "nvidia": {}
}

# 收集结果
for dataset in DATASETS:
    for scene in SCENES[dataset]:
        results[dataset][scene] = {}
        for method in METHODS:
            path = Path(ROOT_DIR) / method / dataset / scene / "results.json"
            if path.exists():
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    # 提取 ours_14000 的指标
                    metrics_data = data.get("ours_14000", {})
                    ssim = metrics_data.get("SSIM", "/")
                    psnr = metrics_data.get("PSNR", "/")
                    lpips_vgg = metrics_data.get("LPIPS-vgg", "/")
                    # 仅记录需要的指标
                    results[dataset][scene][method] = [ssim, psnr, lpips_vgg]
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    results[dataset][scene][method] = ["/", "/", "/"]
            else:
                results[dataset][scene][method] = ["/", "/", "/"]

# 计算平均值（仅对数值）
def safe_avg(values):
    nums = [float(v) for v in values if v != "/"]
    return f"{sum(nums)/len(nums):.6f}" if nums else "/"

# 写入 CSV
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 写表头
    header = ["Dataset/Scene"] + SCENES["nerfie"] + ["avg"]
    writer.writerow(header)

    # 写每个数据集
    for dataset in DATASETS:
        # SSIM 行
        ssim_row = [f"{dataset} - SSIM"]
        ssim_vals = []
        for scene in SCENES[dataset]:
            val = results[dataset][scene]["vggt"][0] if results[dataset][scene]["vggt"][0] != "/" else "/"
            ssim_row.append(val)
            if val != "/":
                ssim_vals.append(float(val))
        avg_ssim = safe_avg(ssim_vals)
        ssim_row.append(avg_ssim)
        writer.writerow(ssim_row)

        # PSNR 行
        psnr_row = [f"{dataset} - PSNR"]
        psnr_vals = []
        for scene in SCENES[dataset]:
            val = results[dataset][scene]["vggt"][1] if results[dataset][scene]["vggt"][1] != "/" else "/"
            psnr_row.append(val)
            if val != "/":
                psnr_vals.append(float(val))
        avg_psnr = safe_avg(psnr_vals)
        psnr_row.append(avg_psnr)
        writer.writerow(psnr_row)

        # LPIPS-vgg 行
        lpips_row = [f"{dataset} - LPIPS-vgg"]
        lpips_vals = []
        for scene in SCENES[dataset]:
            val = results[dataset][scene]["vggt"][2] if results[dataset][scene]["vggt"][2] != "/" else "/"
            lpips_row.append(val)
            if val != "/":
                lpips_vals.append(float(val))
        avg_lpips = safe_avg(lpips_vals)
        lpips_row.append(avg_lpips)
        writer.writerow(lpips_row)

        # Monst3R 行（SSIM, PSNR, LPIPS-vgg）
        monst3r_ssim = [f"Monst3R - SSIM"]
        monst3r_psnr = [f"Monst3R - PSNR"]
        monst3r_lpips = [f"Monst3R - LPIPS-vgg"]
        m_ssim_vals, m_psnr_vals, m_lpips_vals = [], [], []
        for scene in SCENES[dataset]:
            ssim_val = results[dataset][scene]["monst3r"][0]
            psnr_val = results[dataset][scene]["monst3r"][1]
            lpips_val = results[dataset][scene]["monst3r"][2]

            monst3r_ssim.append(ssim_val)
            monst3r_psnr.append(psnr_val)
            monst3r_lpips.append(lpips_val)

            if ssim_val != "/": m_ssim_vals.append(float(ssim_val))
            if psnr_val != "/": m_psnr_vals.append(float(psnr_val))
            if lpips_val != "/": m_lpips_vals.append(float(lpips_val))

        monst3r_ssim.append(safe_avg(m_ssim_vals))
        monst3r_psnr.append(safe_avg(m_psnr_vals))
        monst3r_lpips.append(safe_avg(m_lpips_vals))

        writer.writerow(monst3r_ssim)
        writer.writerow(monst3r_psnr)
        writer.writerow(monst3r_lpips)

        # VGGT-4D 行（即 vggt，已写在前面）
        # 注意：上面已写入 VGGT 的结果，所以这里不需要重复写
        # 但为了清晰，可以加一行分隔
        if dataset == "nerfie":
            writer.writerow([])  # 空行分隔两个数据集

print(f"✅ Results saved to {csv_path}")