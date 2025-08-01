import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import platform

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'  # 리눅스 또는 기타

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 데이터 생성 (Ramsey 진동 + 지수 감쇠)
def ramsey(t, A, f, phi, T2, offset):
    return A * np.cos(2 * np.pi * f * t + phi) * np.exp(-t / T2) + offset

# 가상 데이터
np.random.seed(0)
t_data = np.linspace(0, 10, 60)
signal = ramsey(t_data, 1.0, 1.2, 0, 2.85, 0) + np.random.normal(0, 0.1, len(t_data))

# 피팅
popt, _ = curve_fit(ramsey, t_data, signal, p0=[1, 1, 0, 3, 0])

# 피팅 결과
fit_curve = ramsey(t_data, *popt)
T2_extracted = popt[3]

# 그래프 출력
plt.figure(figsize=(10, 5))
plt.scatter(t_data, signal, label="측정 데이터", color="blue", s=15)
plt.plot(t_data, fit_curve, label=f"피팅: T₂ ≈ {T2_extracted:.2f} μs", color="red", linewidth=2)
plt.title("Ramsey 진동에 따른 위상일관성 & T₂ 추정")
plt.xlabel("시간 (μs)")
plt.ylabel("신호 진폭")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()