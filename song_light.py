import board
import neopixel
from colorsys import hsv_to_rgb
pixels = neopixel.NeoPixel(board.D18, 60)

from realtime_predictor import *

counter = 0

def on_predicted_birds(ensembled_pred):
    result = np.argmax(ensembled_pred)
    label = conf.labels[result]
    p = ensembled_pred[result]
    level = int(p*10) + 1
    global counter
    counter += 1
    if counter % 100 == 0:
        print("test", level, label, p)
        hue = conf.label2int.get(label) / 88
        colourHSV = (hue,1,1)
        colourRGB = hsv_to_rgb(*colourHSV)
        colourRGB = tuple(x*255 for x in colourRGB)
        colourRGB = tuple(map(int, colourRGB))
        print(hue, colourRGB)
        pixel_num = int(counter / 100)
        pixels[pixel_num] = colourRGB

if __name__ == '__main__':
    model = get_model(args.model_pb_graph)
    # file mode
    if args.input_file != '':
        process_file(model, args.input_file)
        my_exit(model)
    # device list display mode
    if args.input < 0:
        print_pyaudio_devices()
        my_exit(model)
    # normal: realtime mode
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    audio = pyaudio.PyAudio()
    stream = audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=44100,
                input=True,
                input_device_index=1,
                frames_per_buffer=conf.rt_chunk_samples,
                start=False,
                stream_callback=callback # uncomment for non_blocking
            )
    # main loop
    stream.start_stream()
    # print(conf.label2int)
    while stream.is_active():
        main_process(model, on_predicted_birds)
        time.sleep(0.001)
    stream.stop_stream()
    stream.close()
    # finish
    audio.terminate()
    my_exit(model)