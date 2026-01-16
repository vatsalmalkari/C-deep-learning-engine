import cv2
import numpy as np
import struct

def relu(x):
    return np.maximum(0, x)

def max_pool(x):
    n, c, h, w = x.shape
    # Simple 2x2 max pool
    out = x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))
    return out

def conv2d(x, w, b):
    # x: (1, in_c, h, w)
    # w: (out_c, in_c, k, k) - weight tensor
    # b: (out_c) - bias tensor
    
    if isinstance(w, int):
        print("Error: Weight w is an int!", w)
        return x
        
    out_c, in_c, k, _ = w.shape
    h_out, w_out = x.shape[2], x.shape[3]
    
    x_pad = np.pad(x, ((0,0), (0,0), (1,1), (1,1)), mode='constant')
    
    output = np.zeros((1, out_c, h_out, w_out))
    
    for i in range(h_out):
        for j in range(w_out):
            patch = x_pad[:, :, i:i+k, j:j+k]
           
            output[:, :, i, j] = np.sum(patch * w, axis=(1, 2, 3)) + b
            
    return output

def linear(x, w, b):
    return np.dot(x, w.T) + b

def load_and_run():
    print("Loading fer_model.bin") # this is the main model i used feel free to use any
    
    tensors = []
    with open("fer_model.bin", "rb") as f:
        
        count_bytes = f.read(8)
        if not count_bytes:
            print("Error: Empty file.")
            return
            
        count = struct.unpack("Q", count_bytes)[0]
        print(f"Found {count} tensors in file.")
        
        for i in range(count):
            # 2. Read data size (size_t = 8 bytes)
            size_bytes = f.read(8)
            data_size = struct.unpack("Q", size_bytes)[0]
            
            # 3. Read float data
            float_bytes = f.read(data_size * 4) # 4 bytes per float
            data = struct.unpack(f"{data_size}f", float_bytes)
            
            arr = np.array(data, dtype=np.float32)
            tensors.append(arr)
            print(f"  Loaded tensor {i}: size={data_size}")

    #reshape - flatten in the binary file
    
    # Conv1 Weights (12 filters, 1 channel, 3x3)
    c1_w = tensors[0].reshape(12, 1, 3, 3)
    c1_b = tensors[1]
    
    # Conv2 Weights (24 filters, 12 channels, 3x3)
    c2_w = tensors[2].reshape(24, 12, 3, 3)
    c2_b = tensors[3]
    
    # FC Weights (7 outputs, input is 24 * 12 * 12)
    fc_w = tensors[4].reshape(7, 3456)
    fc_b = tensors[5]

    #  WEBCAM
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    print("Running")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
           
            img = roi.astype(np.float32) / 255.0
            input_tensor = img.reshape(1, 1, 48, 48)
            
            # --- FORWARD PASS ---
            out = conv2d(input_tensor, c1_w, c1_b)
            out = relu(out)
            out = max_pool(out)
            
            out = conv2d(out, c2_w, c2_b)
            out = relu(out)
            out = max_pool(out)
            
            out = out.reshape(1, -1) # Flatten
            logits = linear(out, fc_w, fc_b)
            
            pred = np.argmax(logits)
            label = emotions[pred]
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        cv2.imshow('Custom C++ Brain', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    load_and_run()