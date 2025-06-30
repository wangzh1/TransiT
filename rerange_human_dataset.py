import os
import shutil

def move_and_rename_subfolders(source_dir, target_dir):
    # 如果目标目录不存在，创建它
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # 遍历源目录下的所有以 'X-bot' 开头的子文件夹
    for dir_name in os.listdir(source_dir):
        if dir_name.startswith('real'):
            x_bot_dir = os.path.join(source_dir, dir_name)
            if os.path.isdir(x_bot_dir):
                # 获取 X-bot 子文件夹内的所有子文件夹
                subfolders = [f for f in os.listdir(x_bot_dir) if os.path.isdir(os.path.join(x_bot_dir, f))]
                for subfolder_name in subfolders:
                    subfolder_path = os.path.join(x_bot_dir, subfolder_name)
                    # 新的子文件夹名称：'X-bot' 文件夹名 + '_' + 原子文件夹名
                    new_name = f"{dir_name}_{subfolder_name}"
                    target_subfolder = os.path.join(target_dir, new_name)
                    # 移动并重命名子文件夹
                    shutil.move(subfolder_path, target_subfolder)
                # 如果需要删除空的 X-bot 文件夹，可以取消下面的注释
                # os.rmdir(x_bot_dir)
    
    print("子文件夹已成功移动并重命名。")

# 使用示例
source_directory = '/home/user/NLOS-Video/sim_results/video/human'  # 请替换为您的源目录路径
target_directory = '/home/user/NLOS-Video/sim_results/video/Xbot'  # 请替换为您的目标目录路径

move_and_rename_subfolders(source_directory, target_directory)
