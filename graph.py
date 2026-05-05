import matplotlib.pyplot as plt

def plot_confidence(timestamps, scores):
    fig, ax = plt.subplots()
    ax.plot(timestamps, scores)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Fake Confidence")
    ax.set_title("Confidence vs Time")
    return fig