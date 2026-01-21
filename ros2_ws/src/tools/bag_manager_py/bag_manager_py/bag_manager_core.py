#!/usr/bin/env python3
import os
import signal
import subprocess
import datetime
import threading
import logging
from typing import List, Tuple, Optional

class BagRecorderCore:
    """ROS 2バッグの録画実行とファイルシステム操作を制御するクラス。"""

    def __init__(self, output_dir: str, topics: List[str], all_topics: bool, 
                 storage_id: str, logger: Optional[logging.Logger] = None):
        """
        Args:
            output_dir (str): 保存先ベースディレクトリ。
            topics (List[str]): 録画対象トピック。
            all_topics (bool): 全トピック録画の有効化フラグ。
            storage_id (str): ストレージ形式 ('mcap' / 'sqlite3')。
            logger (Optional[logging.Logger]): ログ出力用インスタンス。
        """
        self.output_dir = output_dir
        self.topics = topics
        self.all_topics = all_topics
        self.storage_id = storage_id
        self.logger = logger or logging.getLogger(__name__)

        self.session_dir = self._setup_session_dir()
        self.recording_process: Optional[subprocess.Popen] = None
        self.is_recording = False
        self.last_record_dir: Optional[str] = None
        self.lock = threading.Lock()

    def _setup_session_dir(self) -> str:
        """起動時刻に基づいたセッション用ディレクトリを作成する。"""
        now = datetime.datetime.now()
        session_ts = now.strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.output_dir, session_ts)
        os.makedirs(path, exist_ok=True)
        return path

    def start(self) -> Tuple[bool, str]:
        """ros2 bag recordをサブプロセスとして開始する。"""
        with self.lock:
            if self.is_recording:
                return False, "Already recording"

            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            record_dir = os.path.join(self.session_dir, ts)
            
            cmd = self._build_command(record_dir)
            if not cmd:
                return False, "No topics to record"

            try:
                # プロセスグループ制御のため os.setsid を適用
                self.recording_process = subprocess.Popen(cmd, preexec_fn=os.setsid)
                self.is_recording = True
                self.last_record_dir = record_dir
                return True, f"Started: {record_dir}"
            except Exception as e:
                return False, str(e)

    def stop(self) -> Tuple[bool, str]:
        """録画プロセスを終了させる。"""
        with self.lock:
            if not self.is_recording:
                return False, "Not recording"

            self._stop_process_robustly()
            return True, "Stopped"

    def apply_memo(self, memo: str) -> Tuple[bool, str]:
        """
        直近の録画ディレクトリ名に判定ラベルを付与する。

        Args:
            memo (str): 付与する文字列 ('good' または 'bad')。
        """
        with self.lock:
            if self.is_recording:
                return False, "Recording in progress"
            if not self.last_record_dir or not os.path.exists(self.last_record_dir):
                return False, "No target directory"

            src = self.last_record_dir.rstrip('/')
            parent, name = os.path.split(src)
            
            # 既存の判定ラベルを除去
            clean_name = name.replace('_good', '').replace('_bad', '')
            dst_name = f"{clean_name}_{memo.lower()}"
            dst = os.path.join(parent, dst_name)

            try:
                os.rename(src, dst)
                self.last_record_dir = dst
                return True, f"Renamed to: {dst_name}"
            except Exception as e:
                return False, str(e)

    def _build_command(self, record_dir: str) -> Optional[List[str]]:
        """引数に基づいて ros2 bag record コマンドを生成する。"""
        cmd = ['ros2', 'bag', 'record']
        if self.all_topics:
            cmd.append('-a')
        elif self.topics:
            cmd.extend(self.topics)
        else:
            return None
        cmd.extend(['-o', record_dir, '-s', self.storage_id])
        return cmd

    def _stop_process_robustly(self):
        """SIGINTによる正常終了を試み、失敗した場合は段階的に強制終了する。"""
        if self.recording_process and self.recording_process.poll() is None:
            pgid = os.getpgid(self.recording_process.pid)
            try:
                os.killpg(pgid, signal.SIGINT)
                self.recording_process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGTERM)
                self.recording_process.wait(timeout=5.0)
            finally:
                if self.recording_process and self.recording_process.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
                    self.recording_process.wait()
        
        self.recording_process = None
        self.is_recording = False