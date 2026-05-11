import cv2

def try_open_camera(index):
    print(f"\nTrying index {index} with MSMF backend...")
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)

    if not cap.isOpened():
        print(f"❌ Cannot open camera index {index}")
        return False

    ret, frame = cap.read()
    if not ret:
        print(f"⚠️ Opened index {index} but cannot read frame!")
        cap.release()
        return False

    print(f"✅ Camera index {index} WORKING")
    cap.release()
    return True


print("\n======================")
print(" CAMERA DETECTION TEST")
print("======================\n")

available = []

# test only 0–4 safely
for i in range(5):
    if try_open_camera(i):
        available.append(i)

print("\n======================")
print(" AVAILABLE CAMERAS")
print("======================")
print(available)

if not available:
    print("\n❌ No working cameras found.")
    exit()

print("\nSelect a camera index to preview:")
choice = int(input("> "))

cap = cv2.VideoCapture(choice, cv2.CAP_MSMF)

if not cap.isOpened():
    print("❌ Failed to open selected camera.")
    exit()

print("\n🎥 Showing camera. Press 'q' to close.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Cannot grab frame.")
        break

    cv2.imshow("Camera Preview", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
