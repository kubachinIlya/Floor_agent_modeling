

mean_signals = [55, 54, 45, 64, 54, 12, 99]
min_value = min(mean_signals)
max_value = max(mean_signals)

# нормализуем уровень сигнала range( 0 - 1 )
for i in range(0, len(mean_signals)):
    normalized_signal = (mean_signals.__getitem__(i) - min_value) / (max_value - min_value)
    mean_signals[i] = normalized_signal

print(mean_signals)