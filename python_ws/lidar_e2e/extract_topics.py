import argparse
import multiprocessing  
import os               
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


def extract_and_save_per_bag(bag_path, output_dir, scan_topic, cmd_topic):
    """
    単一のrosbagファイルからLaserScanとAckermannDriveStampedデータを抽出・同期する並列ワーカー関数。
    """
    pid = os.getpid()  
    bag_path = Path(bag_path).expanduser().resolve()
    bag_name = bag_path.name
    out_dir = Path(output_dir) / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_data, scan_times = [], []
    cmd_data, cmd_times = [], []

    try:
        with AnyReader([bag_path]) as reader:
            # 指定されたトピックのみをフィルタリング
            connections = [c for c in reader.connections if c.topic in [scan_topic, cmd_topic]]

            for conn, timestamp, raw in reader.messages(connections=connections):
                msg = reader.deserialize(raw, conn.msgtype)

                # 1. LaserScanの処理
                if conn.topic == scan_topic and conn.msgtype == 'sensor_msgs/msg/LaserScan':
                    # 無限遠(inf)などは必要に応じてnp.nan_to_numなどで処理可能
                    scan_data.append(np.array(msg.ranges, dtype=np.float32))
                    scan_times.append(timestamp)

                # 2. AckermannDriveStampedの処理
                elif conn.topic == cmd_topic and conn.msgtype == 'ackermann_msgs/msg/AckermannDriveStamped':
                    # .drive メンバにアクセスして値を取得
                    cmd_data.append(np.array([msg.drive.steering_angle, msg.drive.speed], dtype=np.float32))
                    cmd_times.append(timestamp)

    except Exception as e:
        print(f"[PID:{pid} ERROR] {bag_name}: Failed to read bag file. {e}")
        return

    # データの存在チェック
    if len(scan_data) == 0 or len(cmd_data) == 0:
        print(f'[PID:{pid} WARN] Skipping {bag_name}: insufficient data (scans: {len(scan_data)}, cmds: {len(cmd_data)})')
        return

    # 同期処理のためにNumPy配列化
    scan_times = np.array(scan_times)
    cmd_data, cmd_times = np.array(cmd_data), np.array(cmd_times)

    synced_scans, synced_steers, synced_speeds = [], [], []

    # LaserScanの時刻を基準に、最も近い時刻のコマンドを紐付け
    for i, stime in enumerate(scan_times):
        idx_cmd = np.argmin(np.abs(cmd_times - stime))
        
        synced_scans.append(scan_data[i])
        synced_steers.append(cmd_data[idx_cmd][0])
        synced_speeds.append(cmd_data[idx_cmd][1])

    # データの保存
    np.save(out_dir / 'scans.npy', np.array(synced_scans))
    np.save(out_dir / 'steers.npy', np.array(synced_steers))
    np.save(out_dir / 'speeds.npy', np.array(synced_speeds))

    print(f'[PID:{pid} SAVE] {bag_name}: {len(synced_scans)} samples saved to {out_dir}')

def main():
    parser = argparse.ArgumentParser(description='Extract and synchronize LaserScan and AckermannDriveStamped data from rosbags.')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bags_dir', help='Path to directory containing rosbag folders (searches recursively)')
    group.add_argument('--seq_dirs', nargs='+', help='List of specific sequence directories to process')

    parser.add_argument('--outdir', required=True, help='Output root directory')
    parser.add_argument('--scan_topic', default='/scan', help='LaserScan topic name')
    parser.add_argument('--cmd_topic', default='/jetracer/cmd_drive', help='AckermannDriveStamped topic name')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    bag_dirs = []

    # モードに応じたバッグディレクトリの収集
    if args.bags_dir:
        bags_dir_path = Path(args.bags_dir).expanduser().resolve()
        for p in bags_dir_path.rglob("metadata.yaml"):
            bag_dirs.append(p.parent)
    elif args.seq_dirs:
        for seq_path_str in args.seq_dirs:
            seq_path = Path(seq_path_str).expanduser().resolve()
            if (seq_path / "metadata.yaml").is_file():
                bag_dirs.append(seq_path)

    if not bag_dirs:
        print("[ERROR] No valid rosbag directories found.")
        return

    print(f"[INFO] Found {len(bag_dirs)} rosbag directories to process.")

    # タスクの作成
    tasks = [(p, args.outdir, args.scan_topic, args.cmd_topic) for p in sorted(bag_dirs)]

    # 並列数の設定
    if args.workers:
        num_workers = args.workers
    else:
        num_workers = min(max(1, (os.cpu_count() or 1) - 1), 8)

    print(f"[INFO] Starting parallel processing with {num_workers} workers...")

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(extract_and_save_per_bag, tasks)
        print("[INFO] All processing finished.")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")


if __name__ == '__main__':
    # Linux環境でのマルチプロセス動作の安定化
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    main()