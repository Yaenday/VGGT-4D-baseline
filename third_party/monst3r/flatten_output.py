import os
import shutil
import argparse

def flatten_output_dirs(root_path):
    """
    将 root_path 下所有形如:
        <scene>/output/<scene>/*
    的结构展平为:
        <scene>/output/*

    即把 output 下的冗余同名子目录中的内容移动到 output 本身，然后删除子目录。
    """
    if not os.path.exists(root_path):
        print(f"错误: 路径 '{root_path}' 不存在。")
        return

    if not os.path.isdir(root_path):
        print(f"错误: 路径 '{root_path}' 不是一个目录。")
        return

    # 遍历所有 output 目录
    for dirpath, dirnames, filenames in os.walk(root_path):
        # 只关注名为 'output' 的目录
        if os.path.basename(dirpath) == 'output':
            parent_dir_name = os.path.basename(os.path.dirname(dirpath))  # 父目录名，如 chess4
            redundant_dir = os.path.join(dirpath, parent_dir_name)        # 期望的冗余路径

            if os.path.isdir(redundant_dir):
                print(f"正在处理: {redundant_dir} -> {dirpath}")

                # 移动冗余目录中的所有内容到 output
                for item in os.listdir(redundant_dir):
                    src = os.path.join(redundant_dir, item)
                    dst = os.path.join(dirpath, item)

                    # 防止同名文件冲突
                    if os.path.exists(dst):
                        print(f"  跳过 (已存在): {dst}")
                        continue

                    shutil.move(src, dst)
                    print(f"  移动: {src} -> {dst}")

                # 删除空的冗余目录
                try:
                    os.rmdir(redundant_dir)
                    print(f"  删除空目录: {redundant_dir}")
                except OSError as e:
                    print(f"  删除失败 {redundant_dir}: {e}")
            else:
                print(f"跳过: 无冗余目录 {redundant_dir}")

    print("✅ 所有 output 目录展平完成。")

# === 使用示例 ===
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Flatten output directories.")
    parser.add_argument("--scene_dir", type=str, default="/root/autodl-tmp/hai/VGGT-4D-baseline/data/monst3r",
                        help="Root directory containing scene folders.")
    args = parser.parse_args()

    flatten_output_dirs(args.scene_dir)