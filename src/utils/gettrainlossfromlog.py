import re


log_file = '../../outputs/Isabel_mse_active_random/base100ep/logs/run_logger.log'


train_loss_values = []
test_loss_values = []

train_pattern = r'Average train loss:\s+(\d+\.\d+)'
test_pattern = r'Average Test set loss:\s+(\d+\.\d+)'

with open(log_file, 'r') as file:
    for line in file:
        train_match = re.search(train_pattern, line)
        test_match = re.search(test_pattern, line)

        if train_match:
            train_loss_value = float(train_match.group(1))
            train_loss_values.append(train_loss_value)
        if test_match:
            test_loss_value = float(test_match.group(1))
            test_loss_values.append(test_loss_value)