import os
import glob
import subprocess
import sys
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, ListView, ListItem, Log, Input
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive

# ---------------------------
# コマンド実行
# ---------------------------
def run_command(command):
    return subprocess.run(command, capture_output=True, text=True)

def get_nodes():
    result = run_command(['ros2', 'node', 'list'])
    return [n for n in result.stdout.splitlines() if n]

def get_yaml_files_recursive(directory):
    files = glob.glob(os.path.join(directory, "**/*.yaml"), recursive=True)
    files += glob.glob(os.path.join(directory, "**/*.yml"), recursive=True)
    return sorted(files)

# ---------------------------
# メインアプリ
# ---------------------------
class ROSParamTUI(App):

    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        layout: horizontal;
        height: 1fr;
    }

    ListView {
        border: solid green;
        width: 1fr;
    }

    #log {
        height: 10;
        border: solid yellow;
        overflow-y: auto;
    }

    .title {
        height: 1;
        content-align: center middle;
        text-style: bold;
    }

    #status {
        height: 1;
        border: solid blue;
    }
    """

    current_node = reactive(None)
    base_dir = reactive(os.getcwd())

    # ---------------------------
    # 初期化
    # ---------------------------
    def __init__(self, start_dir=None):
        super().__init__()
        if start_dir and os.path.isdir(start_dir):
            self.base_dir = os.path.abspath(start_dir)

    # ---------------------------
    # UI構築
    # ---------------------------
    def compose(self) -> ComposeResult:
        yield Header()

        # ベースディレクトリ入力欄
        self.dir_input = Input(value=self.base_dir, placeholder="Base Directory")
        yield self.dir_input

        with Horizontal(id="main"):

            with Vertical():
                yield Static("📡 Nodes", classes="title")
                self.node_list = ListView()
                yield self.node_list

            with Vertical():
                yield Static("📂 YAML Files (recursive)", classes="title")
                self.file_list = ListView()
                yield self.file_list

        self.status = Static(id="status")
        yield self.status

        self.log_widget = Log(id="log")
        yield self.log_widget

        yield Footer()

    # ---------------------------
    # 起動時
    # ---------------------------
    def on_mount(self):
        self.refresh_nodes()
        self.refresh_files()
        self.update_status()

    # ---------------------------
    # ログ
    # ---------------------------
    def log_info(self, msg):
        self.log_widget.write_line(f"[white]{msg}[/]")
        self.log_widget.scroll_end()

    def log_success(self, msg):
        self.log_widget.write_line(f"[green]{msg}[/]")
        self.log_widget.scroll_end()

    def log_error(self, msg):
        self.log_widget.write_line(f"[red]{msg}[/]")
        self.log_widget.scroll_end()

    # ---------------------------
    # ステータス表示
    # ---------------------------
    def update_status(self):
        node = self.current_node or "未選択"
        self.status.update(f"Node: {node} | BaseDir: {self.base_dir}")

    # ---------------------------
    # ノード更新
    # ---------------------------
    def refresh_nodes(self):
        self.node_list.clear()
        for node in get_nodes():
            self.node_list.append(ListItem(Static(node)))

    # ---------------------------
    # ファイル更新（再帰）
    # ---------------------------
    def refresh_files(self):
        self.file_list.clear()

        files = get_yaml_files_recursive(self.base_dir)

        if not files:
            self.log_info("YAMLが見つかりません")
            return

        for f in files:
            display = os.path.relpath(f, self.base_dir)
            self.file_list.append(ListItem(Static(display)))

    # ---------------------------
    # 入力欄変更（Enterで反映）
    # ---------------------------
    def on_input_submitted(self, event: Input.Submitted):
        path = event.value.strip()

        if not os.path.isdir(path):
            self.log_error("無効なディレクトリ")
            return

        self.base_dir = os.path.abspath(path)
        self.log_success(f"📂 BaseDir変更: {self.base_dir}")

        self.refresh_files()
        self.update_status()

    # ---------------------------
    # 選択イベント
    # ---------------------------
    def on_list_view_selected(self, event: ListView.Selected):

        # ノード
        if event.list_view is self.node_list:
            self.current_node = str(event.item.query_one(Static).renderable)
            self.log_success(f"🎯 Node selected: {self.current_node}")
            self.update_status()

        # YAML
        elif event.list_view is self.file_list:
            filename = str(event.item.query_one(Static).renderable)
            self.load_param(filename)

    # ---------------------------
    # ロード
    # ---------------------------
    def load_param(self, relative_path):

        if not self.current_node:
            self.log_error("ノード未選択")
            return

        full_path = os.path.join(self.base_dir, relative_path)

        self.log_info(f"⏳ Loading: {relative_path}")

        result = run_command(['ros2', 'param', 'load', self.current_node, full_path])

        if result.returncode == 0:
            self.log_success("✅ 成功")
        else:
            self.log_error("❌ 失敗")
            self.log_error(result.stderr.strip())

    # ---------------------------
    # キーバインド
    # ---------------------------
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reload", "Reload"),
    ]

    def action_reload(self):
        self.refresh_nodes()
        self.refresh_files()
        self.log_info("🔄 Reload")


# ---------------------------
# エントリポイント
# ---------------------------
if __name__ == "__main__":
    start_dir = sys.argv[1] if len(sys.argv) > 1 else None
    ROSParamTUI(start_dir=start_dir).run()