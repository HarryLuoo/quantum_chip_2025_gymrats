import numpy as np
import matplotlib.pyplot as plt

# 큐비트 수
n = 3

# 입력 자기장: Q1만 자극
B_input = np.zeros(n)
B_input[1] = 1.0  # Q1에 자기장

# crosstalk 계수 (인접 큐비트 간 영향)
crosstalk_coeff = 0.3

# crosstalk 행렬 (Q1 → Q0, Q2)
C = np.eye(n)
C[0, 1] = crosstalk_coeff
C[2, 1] = crosstalk_coeff

# crosstalk 적용 후 자기장
B_effective = C @ B_input

# 보정 자기장: Q0, Q2에 -crosstalk 방향으로 인가
B_correction = np.zeros(n)
B_correction[0] = -C[0,1] * B_input[1]
B_correction[2] = -C[2,1] * B_input[1]

# 최종 자기장 (보정 포함)
B_total_corrected = B_effective + B_correction

# 시각화
labels = ['Q0', 'Q1', 'Q2']
x = np.arange(n)
bar_width = 0.25

plt.figure(figsize=(10, 6))

# 각 단계의 막대 그래프
plt.bar(x - bar_width, B_input, width=bar_width, label='Input B', color='skyblue')
plt.bar(x, B_effective, width=bar_width, label='After Crosstalk', color='orange')
plt.bar(x + bar_width, B_total_corrected, width=bar_width, label='Corrected B', color='seagreen')

# 시각적 요소
plt.xticks(x, labels, fontsize=12)
plt.ylabel('Magnetic Field Strength', fontsize=12)
plt.title('Qubit Crosstalk Correction (Linear Layout)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# y축 스케일 자동 조정
y_max = max(np.max(np.abs(B_input)), np.max(np.abs(B_effective)), np.max(np.abs(B_total_corrected)))
plt.ylim(-y_max * 1.1, y_max * 1.1)

plt.tight_layout()
plt.show()
