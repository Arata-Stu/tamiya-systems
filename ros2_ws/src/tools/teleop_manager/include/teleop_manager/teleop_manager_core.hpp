#ifndef TELEOP_MANAGER_CORE_HPP_
#define TELEOP_MANAGER_CORE_HPP_

#include <optional>
#include <string>
#include <vector>

class TeleopManagerCore {
public:
  struct Config {
    double speed_scale;
    double steer_scale;
    int joy_button_idx;
    int ack_button_idx;
    int start_button_idx;
    int stop_button_idx;
    int good_button_idx;
    int bad_button_idx;
    int dpad_lr_axis_idx;
    int dpad_ud_axis_idx;
    int axis_speed_idx;
    int axis_steer_idx;
  };

  struct DriveCommand {
    double acceleration;
    double steering_angle;
    double steering_velocity;
  };

  explicit TeleopManagerCore(const Config &config);

  void update_joy_input(const std::vector<float> &axes,
                        const std::vector<int> &buttons);

  DriveCommand calculate_drive_command(bool is_timeout, double autonomy_speed,
                                       double autonomy_steer);

  bool pop_start_requested();
  bool pop_stop_requested();
  std::optional<std::string> pop_memo_requested();
  bool pop_steer_inc_requested();
  bool pop_steer_dec_requested();
  bool pop_speed_inc_requested();
  bool pop_speed_dec_requested();

private:
  bool check_button_press(bool curr, bool &prev_flag);

  Config config_;
  bool joy_active_{false};
  bool ack_active_{false};
  double joy_speed_{0.0};
  double joy_steer_{0.0};

  bool prev_start_{false}, prev_stop_{false};
  bool prev_good_{false}, prev_bad_{false};
  bool prev_st_inc_{false}, prev_st_dec_{false};
  bool prev_sp_inc_{false}, prev_sp_dec_{false};

  bool start_req_{false}, stop_req_{false};
  bool st_inc_req_{false}, st_dec_req_{false};
  bool sp_inc_req_{false}, sp_dec_req_{false};
  std::optional<std::string> memo_req_{std::nullopt};
};

#endif