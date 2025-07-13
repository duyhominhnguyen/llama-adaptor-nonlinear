import matplotlib.pyplot as plt

loss_non_linear = None # loss value in training log located in checkpoint folder
loss_linear = None # loss value in training log located in checkpoint folder
loss_random = None # loss value in training log located in checkpoint folder
iter_list = [0, 1, 2, 3, 4]
plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(8, 6))
plt.plot(iter_list, loss_linear, label='Linear prompt', marker='o', linestyle='-.', color='blue')
plt.plot(iter_list, loss_random, label='Random prompt', marker='^', linestyle='--', color='green')
plt.plot(iter_list, loss_non_linear, label='Non-Linear prompt', marker='s', linestyle='-', color='red')

for i, value in enumerate(loss_non_linear):
    plt.text(iter_list[i], loss_non_linear[i]-0.023, f'{value:.3f}', color='purple', fontsize=12, ha='center')

for i, value in enumerate(loss_random):
    if i == 0:
        plt.text(iter_list[i], 1.5, f'{value:.3f}', color='green', fontsize=12, ha='center')
    else:
        plt.text(iter_list[i], loss_random[i]+0.023, f'{value:.3f}', color='green', fontsize=12, ha='center')

plt.ylim(1.02, 1.55)
# Add labels and title
plt.xlabel('Epoch (x)', fontsize=16)
plt.ylabel('Loss Value', fontsize=16)
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png', format='png', dpi=300)