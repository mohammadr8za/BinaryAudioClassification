from tensorboard import program
tracking_address = r'D:\mreza\TestProjects\Python\BinaryAudioClassification\scripts\runs'

if __name__ == '__main__':
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

