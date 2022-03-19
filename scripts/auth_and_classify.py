import argparse
import os
import queue
import struct
import sys
import threading
import time
from matplotlib import colorbar
import numpy as np
import sklearn
import pandas as pd
import datetime
from sklearn.preprocessing import OrdinalEncoder

import matplotlib.pyplot as plt
import serial
import zmq

import urllib.request

URL = "https://perceval.elen.ucl.ac.be/lelec2103/push/{key}/{answer}"
ANSWER_TYPES = ('fire', 'birds', 'helicopter', 'chainsaw', 'handsaw')

GROUP_KEY = "85ba068f39"

from classify import features, modelling
import packet

import dataset
import classificationModel


# For plotting only
MELVEC_MIN = 0.0
MELVEC_MAX = 1.0

PRINT_PREFIX = "DF:HEX:"
def parse_packet(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX):])
    else:
        return None

def is_file(path):
    if os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError(path)

categories = ['chainsaw', 'chirping_birds','crackling_fire','hand_saw','helicopter','nothing']
audio_path = '../ESC-50/audio/'
meta_path = '../ESC-50/meta/'
nothing_path = './room_sounds/'

data = pd.read_csv(meta_path + 'esc50.csv')

## Remove rows where the category does not belong to categories
data = data[data.category.isin(categories)]

re_encoder = OrdinalEncoder(dtype=np.long)
re_encoder.fit(data[["category"]])
data[["category"]] = re_encoder.transform(data[["category"]])

print(re_encoder.inverse_transform(np.array([0,1,2,3,4]).reshape(-1,1)))

ds = dataset.SoundDS(data,audio_path,False)
model = classificationModel.loadModel("../myModel.pt")


description = """
Python tool to perform the parsing of packets from the MCU, their
authentication and the classification of the transmitted feature vectors.

By default, this tool only reads packets from the ZMQ protocol. The different
features (authentification, classification, ...) must individually be enabled.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', action="store", type=str,
                help='input stream to read from (either file or serial device, default: zmq GNU Radio interface')
    parser.add_argument('--output', action="store", type=str,
                help='save the received packets at the given location')
    parser.add_argument('--authenticate', action="store_true",
                help='enable the authentification')
    parser.add_argument('--display', action="store_true",
                help='display the received feature vectors')
    parser.add_argument('--classifier', action="store", type=is_file,
                help='path to a classifier model')
    parser.add_argument('-q', '--quiet', action="store_true",
                help='Do not display all input lines (for file or serial input)')

    args = parser.parse_args()

    auth_key = 16*b'\0'
    print('key:', auth_key.hex())
    unwrapper = packet.PacketUnwrapper(
            key=auth_key,
            allowed_senders=[0,],
            authenticate=args.authenticate
            )

    if args.output:
        f_out = open(args.output, 'w')

    if args.classifier:
        # Load classifier
        model = classificationModel.loadModel(args.classifier)

    if args.input == None:
        # Read messages from zmq GNU Radio interface
        def reader():
            context = zmq.Context()
            socket = context.socket(zmq.SUB)

            socket.setsockopt(zmq.SUBSCRIBE, b'')
            socket.setsockopt(zmq.CONFLATE, 1)  # last msg only.

            socket.connect('tcp://127.0.0.1:10000')
            while True:
                msg = socket.recv(2*features.MELVEC_LENGTH*features.N_MELVECS)
                if args.output:
                    f_out.write(PRINT_PREFIX + msg.hex() + '\n')
                yield msg

    elif args.input.startswith("/dev/tty"):
        # Read messages from serial interface
        def reader():
            ser = serial.Serial(port=args.input, baudrate=115200)
            ser.reset_input_buffer()
            ser.read_until(b'\n')
            while True:
                line = ser.read_until(b'\n').decode("ascii").strip()
                if args.output:
                    f_out.write(line+'\n')
                if not args.quiet:
                    print('#', line)
                try:
                    packet = parse_packet(line)
                except ValueError:
                    print("Warning: invalid packet line:", line)
                else:
                    if packet is not None:
                        yield packet

    else:
        # Read messages from file
        def reader():
            f = open(args.input)
            for line in f.readlines():
                line = line.strip()
                if args.output:
                    f_out.write(line+'\n')
                if not args.quiet:
                    print('#', line)
                packet = parse_packet(line)
                if packet is not None:
                    yield packet
                    if args.display:
                        input("Press ENTER to process next packet")

    if args.display:
        plot_queue = queue.Queue(maxsize=1)
        def plotter():
            while True:
                try:
                    melvecs = plot_queue.get_nowait()
                except queue.Empty:
                    pass
                else:
                    im = plt.imshow(
                            melvecs,
                            aspect="auto",
                            cmap='jet',
                            origin='lower'
                            )
                    cb = plt.colorbar()
                plt.pause(0.02)
                try:
                    cb.remove()
                except:
                    pass

        plot_thread = threading.Thread(target=plotter)
        plot_thread.setDaemon(True)
        plot_thread.start()

    input_stream = reader()
    i = 0
    for msg in input_stream:
        try:
            sender, payload = unwrapper.unwrap_packet(msg)
        except packet.InvalidPacket as e:
            print("Invalid packet received:", e.args[0])
            print("\t", PRINT_PREFIX + msg.hex())
            continue

        print(f"From {sender}: packet received", payload.hex())
        melvecs = features.payload2melvecs(payload)

        if args.classifier:
            classes = ["fire", "birds", "helicopter", "chainsaw", "handsaw"]
            melvecs = melvecs/np.amax(melvecs)
            classified = classificationModel.predict(melvecs,ds,model)
            print(i,classified,categories[classified.numpy()],datetime.datetime.now())
            try:
                urllib.request.urlopen(URL.format(key=GROUP_KEY, answer=categories[classified.numpy()]))
            except urllib.error.HTTPError as e:
                print(e)

            np.savetxt("measurements/" + str(i) + ".txt",melvecs)            # print("Sound class:", classes[sound_class[0]])
            i+=1

        if args.display:
            try:
                plot_queue.put_nowait(melvecs)
            except queue.Full:
                pass

