import logging
from typing import Optional
from jetracer.nvidia_racecar import NvidiaRacecar

class JetRacerCore:
    """JetRacerの実機制御と計算ロジックを管理するクラス。"""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Args:
            logger (Optional[logging.Logger]): ログ出力用インスタンス。
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # パラメータの内部保持
        self.params = {
            "throttle_inversion": False,
            "steering_inversion": False,
            "steering_offset": 0.0,
            "throttle_offset": 0.0,
            "throttle_gain": 1.0,
            "steering_gain": 1.0,
        }

        try:
            self.car = NvidiaRacecar()
            self.stop()
        except Exception as e:
            self.logger.error(f"Failed to initialize NvidiaRacecar: {e}")
            raise e

    def set_drive(self, speed: float, steering_angle: float):
        """
        計算式に基づき、値を正規化して実機に指令を送る。

        $throttle = (speed \times gain) + offset$
        $steering = (angle \times gain) + offset$

        Args:
            speed (float): 指令速度。
            steering_angle (float): 指令ステアリング角。
        """
        # スロットル計算
        throttle = (speed * self.params["throttle_gain"]) + self.params["throttle_offset"]
        if self.params["throttle_inversion"]:
            throttle *= -1.0
        throttle = max(min(throttle, 1.0), -1.0)

        # ステアリング計算
        steering = (steering_angle * self.params["steering_gain"]) + self.params["steering_offset"]
        if self.params["steering_inversion"]:
            steering *= -1.0
        steering = max(min(steering, 1.0), -1.0)

        self.car.throttle = float(throttle)
        self.car.steering = float(steering)

    def stop(self):
        """車両の出力をゼロにして停止させる。"""
        self.car.throttle = 0.0
        self.car.steering = 0.0

    def update_param(self, name: str, value: any):
        """
        内部パラメータを更新する。

        Args:
            name (str): パラメータ名。
            value (any): 更新する値。
        """
        if name in self.params:
            self.params[name] = value