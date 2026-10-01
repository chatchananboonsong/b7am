# เขียนโค้ดควบคุม Robomaster EP โดยใช้ Python 3.8 
ให้สร้าง .venv และติดตั้ง library ให้เรียบร้อย

แหล่งข้อมูลอ้างอิง: https://github.com/dji-sdk/RoboMaster-SDK.git https://www.dji.com/robomaster-ep

## สิ่งที่มี: 
- ToF sensor
- Sharp sensor: Datasheet อยู่ที่ ((../../Downloads/GP2Y0A41SK0F.pdf)) ระยะประมาณ 4 - 30 cm ตัวด้านซ้ายอยู่ id1 port1, ตัวด้านขวาอยู่ id2, port2
- IMU
- ESC
- Position
- Attitude
- Status
- Gripper

## ขนาดสนาม:
- ช่องสนามเป็น Grid ขนาดประมาณ 60*60 cm
- ความกว้างกำแพงประมาณ 7.5 cm 

## ขนาดของหุ่น:
- ความยาวประมาณ 33 cm (ไม่รวม Gripper)
- ความยาวประมาณ 38 cm (รวม Gripper)
- ความยาวประมาณ 52 cm (Gripper ยืดสุด ตั้งแต่ท้ายหุ่นจนถึงปลายสุดของ Gripper ที่ยืดไป)
- ความกว้างประมาณ 25 cm

## สิ่งที่ต้องทำ:
1. Calibrate
    - Sharp: รับค่ามาเทียบระยะในโลกจริงแล้วมาพล็อตกราฟหาสมการ
    - ToF: รับค่าเทียบกับโลกจริง
    - Gripper: วัดค่าในโลกจริงและปรับค่า
2. ทำ Multi-threading แบ่งเป็น 2 ตัว
    1. เก็บค่า Sensor รวม Filtering ทำให้ข้อมูลพร้อมใช้งาน เพื่อไปทำ Mapping ที่กำลังเดินอยู่ และเนำไปวิเคราะห์ข้อมูลตอนรันจบทั้งหมด (ใช้เก็บข้อมูลและวิเคราะห์)
    2. ใช้รันหุ่นยนต์ ดึงข้อมูลจาก thread ที่ 1 เพื่อไม่ให้ดึงซ้ำซ้อน
3. ทำระบบเดินเป็นการเดินทีละ Grid (PID control)
    - พยายามให้หุ่นยนต์อยู่ตรงกลางระหว่างกำแพง (ดึงค่าจาก thread 1) ปรับการเคลื่อนที่โดยใช้การเคลื่อนที่แกน y
        1. มีกำแพงข้างหน้า: เอาระยะจาก ToF มาวัดว่าหุ่นยนต์อยู่ตรงกลางของ Grid
            1. มีกำแพง 2 ข้าง: Sharp |L - R| < 2
            2. มีกำแพงแค่ข้างซ้าย: Sharp L +- 2 จากค่าปกติที่วัดได้ว่าหุ่นยนต์อยู่ตรงกลาง
            3. มีกำแพงแค่ข้างขวา: Sharp R +- 2 จากค่าปกติที่วัดได้ว่าหุ่นยนต์อยู่ตรงกลาง
            4. ไม่มีกำแพง
        2. ไม่มีกำแพงข้างหน้า
            1. มีกำแพง 2 ข้าง: Sharp |L - R| < 2
            2. มีกำแพงแค่ข้างซ้าย: Sharp L +- 2 จากค่าปกติที่วัดได้ว่าหุ่นยนต์อยู่ตรงกลาง
            3. มีกำแพงแค่ข้างขวา: Sharp R +- 2 จากค่าปกติที่วัดได้ว่าหุ่นยนต์อยู่ตรงกลาง
            4. ไม่มีกำแพง
4. ทำระบบ Gripper วางของ 
    - ให้วางของได้ทุกจุด โดยลองรันหุ่นเพื่อเก็บค่าผิดพลาดของหุ่นแล้วมาปรับกับ code เพื่อให้วางตรงกลางจุดพอดี
    1. มีกำแพงข้างหน้า: เอาระยะจาก ToF มาวัดว่าหุ่นยนต์อยู่ตรงกลางของ Grid
        1. มีกำแพง 2 ข้าง: Sharp |L - R| < 2
        2. มีกำแพงแค่ข้างซ้าย: Sharp L +- 2 จากค่าปกติที่วัดได้ว่าหุ่นยนต์อยู่ตรงกลาง
        3. มีกำแพงแค่ข้างขวา: Sharp R +- 2 จากค่าปกติที่วัดได้ว่าหุ่นยนต์อยู่ตรงกลาง
        4. ไม่มีกำแพง
    2. ไม่มีกำแพงข้างหน้า
        1. มีกำแพง 2 ข้าง: Sharp |L - R| < 2
        2. มีกำแพงแค่ข้างซ้าย: Sharp L +- 2 จากค่าปกติที่วัดได้ว่าหุ่นยนต์อยู่ตรงกลาง
        3. มีกำแพงแค่ข้างขวา: Sharp R +- 2 จากค่าปกติที่วัดได้ว่าหุ่นยนต์อยู่ตรงกลาง
        4. ไม่มีกำแพง
5. รับข้อมูลสภาพแวดล้อม โดยใช้ map.py จะได้เป็นไฟล์ .json 
    - ทำให้หุ่นรองรับกับข้อมูลที่ export จาก robot_map_plan.json (ตอนใช้งานจริงเริ่มแรกจะให้หุ่นหันหน้าไปที่ทางเดินตลอด ไม่จำเป็นต้องหมุนหุ่นตอนเริ่ม)

---

## 📌 Implementation Reference (แผนผังเอกสารและซอร์สโค้ด)

| ข้อกำหนด (Requirement) | เอกสารอ้างอิง | ซอร์สโค้ดหลัก | คำสั่งหลักผ่าน `main.py` |
| :--- | :--- | :--- | :--- |
| **1. Calibrate (Sharp, ToF, Gripper)** | [docs/STEP1_CALIBRATION.md](docs/STEP1_CALIBRATION.md) | [src/calibrate.py](src/calibrate.py) | `.venv/bin/python main.py calibrate fit data/calibration_measurements.csv` |
| **2. Multi-threading (2 Threads + Telemetry)** | [docs/STEP2_MULTITHREADING.md](docs/STEP2_MULTITHREADING.md) | [src/sensor_pipeline.py](src/sensor_pipeline.py), [src/telemetry.py](src/telemetry.py) | `.venv/bin/python main.py monitor --conn-type ap` |
| **3. เดินทีละ Grid + PID Centering (8 Cases)** | [docs/STEP3_PID.md](docs/STEP3_PID.md) | [src/pid_controller.py](src/pid_controller.py), [src/robot_controller.py](src/robot_controller.py) | `.venv/bin/python main.py step-test --cells 1 --conn-type ap` |
| **4. Gripper วางของ** | *(Next Step)* | [src/robot_controller.py](src/robot_controller.py) | `.venv/bin/python main.py run --conn-type ap` |
| **5. แผนที่และคำสั่ง JSON** | [README.md](README.md) | [src/map_planner.py](src/map_planner.py) | `.venv/bin/python main.py map` |



python explore.py --conn-type ap --grid 6x6 --cell-size-cm 60
python explore.py --conn-type ap --grid 6x6 --cell-size-cm 60 --start 1,5