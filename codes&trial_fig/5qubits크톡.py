import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 설정: 큐비트 수
n_qubits = 5

# Crosstalk Matrix 생성 (대각: 0.95, 인접 비대각: 0.025, 나머지: 0.0125)
def generate_crosstalk_matrix(n, diag=0.95, offdiag=0.025):
    C = np.eye(n) * diag
    for i in range(n):
        for j in range(n):
            if i != j:
                distance = abs(i - j)
                C[i][j] = offdiag / distance  # 가까울수록 영향 크다
    # normalize each row to sum to 1 (stochastic matrix)
    C = C / C.sum(axis=1, keepdims=True)
    return C

# 시뮬레이션: 이상적 상태에 crosstalk 반영
def apply_crosstalk(C, ideal_state):
    return np.dot(C, ideal_state)

# 예시: 모든 큐비트를 |1>로 측정하려고 함
ideal_state = np.array([1, 1, 1, 1, 1], dtype=float)

# Crosstalk matrix 생성 및 적용
C = generate_crosstalk_matrix(n_qubits)
measured_state = apply_crosstalk(C, ideal_state)

# 결과 시각화
plt.figure(figsize=(12, 4))

# Crosstalk matrix
plt.subplot(1, 2, 1)
sns.heatmap(C, annot=True, cmap='coolwarm', fmt=".3f")
plt.title("Crosstalk Matrix (5 Qubits)")
plt.xlabel("Target Qubit")
plt.ylabel("Source Qubit")

# 이상적 vs 측정된 상태 비교
plt.subplot(1, 2, 2)
x = np.arange(n_qubits)
plt.bar(x - 0.15, ideal_state, width=0.3, label="Ideal")
plt.bar(x + 0.15, measured_state, width=0.3, label="Measured w/ Crosstalk")
plt.xticks(x)
plt.title("Ideal vs Measured State (Crosstalk Effect)")
plt.xlabel("Qubit Index")
plt.ylabel("Excitation Level")
plt.legend()

plt.tight_layout()
plt.show()