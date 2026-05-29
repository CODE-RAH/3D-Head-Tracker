"""
این کد توسط امیرفرخانی موسس کدراه پیاده سازی شده
همکار و اساتید (تقلبی) قبل از برداشتن سورس کد
و تموم کردن کار به اسم خودتون ذکر منبع کنید
در غیر اینصورت رضایتی در کار نیست و اگه متوجه بشیم عواقب داره
این کد اولین باره داخل اینترنت پخش میشه
ویدیو مشابه داخل سطح نت ببینیم و ذکر منبع نکرده باشین عواقبش پای خودتون
"""
import sys
import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image, ImageSequence

from simplepbr import init as pbr_init

from direct.showbase.ShowBase import ShowBase
from direct.actor.Actor import Actor
from panda3d.core import (
    AmbientLight, DirectionalLight, Vec4, loadPrcFileData,
    CardMaker, Texture
)

# ---------------------------------------------------------
# تنظیمات پنجره Panda3D
# ---------------------------------------------------------
loadPrcFileData("", "window-title Character Head Tracking")
loadPrcFileData("", "win-size 800 600")
loadPrcFileData("", "sync-video #t")


def clamp(v, v_min, v_max):
    """محدود کردن مقدار بین بازه مشخص (برای طبیعی شدن گردن)."""
    return max(min(v, v_max), v_min)


class SimpleHeadTracker(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # فعال سازی PBR برای GLB
        pbr_init()

        self.disableMouse()
        self.setBackgroundColor(1, 1, 1)

        # ---------------------------------------------------------
        # پارامترهای حرکتی و اسموث
        # ---------------------------------------------------------
        self.SMOOTH = 0.15        # نرم بودن حرکت
        self.YAW_SENS = -400.0    # حساسیت چرخش چپ / راست
        self.PITCH_SENS = -800.0  # حساسیت بالا / پایین
        self.ROLL_SENS = -80.0    # حساسیت کج شدن سر

        # وضعیت فعلی سر بعد از فیلتر اسموث
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        # ---------------- کالیبراسیون خودکار ----------------
        self.calib_frames = 0          # چند فریم برای کالیبراسیون
        self.CALIB_MAX_FRAMES = 60     # حدوداً ۲ ثانیه
        self.yaw_offset = 0.0
        self.pitch_offset = 0.0
        self.roll_offset = 0.0
        self.calibrated = False

        # ---------------------------------------------------------
        # بارگذاری مدل 3D
        # ---------------------------------------------------------
        try:
            self.actor = Actor("destructor_head_model.glb")
            self.actor.reparentTo(self.render)
            self.actor.setScale(1)
            self.actor.setPos(-0.5, 1, -0.3)

            # اگر دیدی کل بدن برعکس یا پشت به دوربین بود، این خط رو آزاد/تغییر بده:
            # self.actor.setH(180)

            # رنگ پوست روشن‌تر
            self.actor.setColor(1, 1, 1, 1)
            self.actor.setColorScale(1.8, 1.5, 1.4, 1)

        except Exception as e:
            print("❌ Error loading model:", e)
            sys.exit()

        # ---------------------------------------------------------
        # دوربین صحنه
        # ---------------------------------------------------------
        self.camera.setPos(0, -3.5, 0.55)
        self.camera.lookAt(0, 0, 0.55)

        # ---------------------------------------------------------
        # پیدا کردن استخوان سر / گردن
        # ---------------------------------------------------------
        self.head_bone = self.find_head_bone()

        # ---------------------------------------------------------
        # نورپردازی
        # ---------------------------------------------------------
        self.setup_lights()

        # ---------------------------------------------------------
        # mediapipe FaceMesh
        # ---------------------------------------------------------
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # وب‌کم
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # ---------------------------------------------------------
        # GIF — بارگذاری + نمایش
        # ---------------------------------------------------------
        self.gif_frames = []
        self.gif_index = 0
        self.gif_fps = 25
        self.last_gif_time = 0.0

        self.load_gif("gif.gif")
        self.create_gif_plane()

        # ---------------------------------------------------------
        # تسک‌های آپدیت
        # ---------------------------------------------------------
        self.taskMgr.add(self.update_loop, "UpdateLoop")
        self.taskMgr.add(self.update_gif, "UpdateGIF")

    # ---------------------------------------------------------
    # انتخاب استخوان سر
    # ---------------------------------------------------------
    def find_head_bone(self):
        priority = [
            "mixamorig:Head", "Head", "head", "def_head",
            "mixamorig:Neck", "Neck", "neck"
        ]

        joints = self.actor.getJoints()

        for name in priority:
            for j in joints:
                if name in j.getName():
                    print("✅ Found bone:", j.getName())
                    return self.actor.controlJoint(None, "modelRoot", j.getName())

        print("⚠️ No head bone found.")
        return None

    # ---------------------------------------------------------
    # نورپردازی
    # ---------------------------------------------------------
    def setup_lights(self):
        al = AmbientLight("al")
        al.setColor(Vec4(0.9, 0.9, 0.9, 1))
        self.render.setLight(self.render.attachNewNode(al))

        dl = DirectionalLight("dl")
        dl.setColor(Vec4(1.1, 1.1, 1.1, 1))
        dln = self.render.attachNewNode(dl)
        dln.setHpr(0, -60, 0)
        self.render.setLight(dln)

    # ============================================================
    # بارگذاری GIF
    # ============================================================
    def load_gif(self, path):
        if not os.path.exists(path):
            print("❌ GIF not found:", path)
            return

        frames = Image.open(path)
        print("🎞 Extracting GIF frames...")

        for frame in ImageSequence.Iterator(frames):
            frame = frame.convert("RGBA")
            tex = Texture()
            tex.setup2dTexture(
                frame.width, frame.height,
                Texture.T_unsigned_byte, Texture.F_rgba
            )
            tex.setRamImage(frame.tobytes())
            self.gif_frames.append(tex)

        print("✅ Loaded", len(self.gif_frames), "frames")

    # ------------------------------------------------------------
    # ساخت Plane نمایش GIF
    # ------------------------------------------------------------
    def create_gif_plane(self):
        cm = CardMaker("gif_bg")
        cm.setFrame(-4, 4, -3, 3)
        self.gif_plane = self.render.attachNewNode(cm.generate())
        self.gif_plane.setPos(0, 5, 0)
        self.gif_plane.setDepthWrite(False)

        # اگر گیف برعکس بود، این دو خط رو تنظیم کن
        self.gif_plane.setP(180)
        self.gif_plane.setScale(-1, 1, 1)

    # ------------------------------------------------------------
    # پخش GIF
    # ------------------------------------------------------------
    def update_gif(self, task):
        if not self.gif_frames:
            return task.cont

        now = task.time
        if now - self.last_gif_time > 1.0 / self.gif_fps:
            self.gif_index = (self.gif_index + 1) % len(self.gif_frames)
            self.gif_plane.setTexture(self.gif_frames[self.gif_index], 1)
            self.last_gif_time = now

        return task.cont

    # ---------------------------------------------------------
    # حلقه اصلی: خواندن وب‌کم و چرخاندن سر مدل
    # ---------------------------------------------------------
    def update_loop(self, task):

        ret, frame = self.cap.read()
        if not ret:
            return task.cont

        # اگر دوست داری مثل آینه باشه، این خط رو فعال کن:
        # frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks and self.head_bone:

            lm = results.multi_face_landmarks[0].landmark

            # نقاط مهم صورت
            nose = lm[1]
            left_eye = lm[159]     # بالا چشم چپ
            right_eye = lm[386]    # بالا چشم راست
            left_ear = lm[234]
            right_ear = lm[454]

            # ---------------- YAW (چپ / راست) ----------------
            face_center_x = (left_ear.x + right_ear.x) / 2.0
            yaw_raw = (nose.x - face_center_x) * self.YAW_SENS

            # ---------------- PITCH (بالا / پایین) ----------------
            eyes_center_y = (left_eye.y + right_eye.y) / 2.0
            pitch_raw = (nose.y - eyes_center_y) * self.PITCH_SENS

            # ---------------- ROLL (کج شدن سر) ----------------
            dx = right_eye.x - left_eye.x
            dy = right_eye.y - left_eye.y
            roll_angle = np.degrees(np.arctan2(dy, dx))
            roll_raw = roll_angle * (self.ROLL_SENS / 80.0)

            # ---------------- کالیبراسیون خودکار ----------------
            if not self.calibrated and self.calib_frames < self.CALIB_MAX_FRAMES:
                # فرض می‌کنیم تو چند فریم اول تقریباً رو‌به‌رو نگاه می‌کنی
                self.yaw_offset = (self.yaw_offset * self.calib_frames + yaw_raw) / (self.calib_frames + 1)
                self.pitch_offset = (self.pitch_offset * self.calib_frames + pitch_raw) / (self.calib_frames + 1)
                self.roll_offset = (self.roll_offset * self.calib_frames + roll_raw) / (self.calib_frames + 1)
                self.calib_frames += 1

                if self.calib_frames == self.CALIB_MAX_FRAMES:
                    self.calibrated = True
                    print("✅ Head calibration done.")
            # بعد از کالیبراسیون، مقدار خام را نسبت به Offset صفر می‌کنیم
            yaw_corr = yaw_raw - self.yaw_offset
            pitch_corr = pitch_raw - self.pitch_offset
            roll_corr = roll_raw - self.roll_offset

            # ---------------- اسموث کردن ----------------
            self.yaw += (yaw_corr - self.yaw) * self.SMOOTH
            self.pitch += (pitch_corr - self.pitch) * self.SMOOTH
            self.roll += (roll_corr - self.roll) * self.SMOOTH

            # ---------------- محدودیت‌های انسانی ----------------
            yaw_clamped = clamp(self.yaw, -60, 60)
            pitch_clamped = clamp(self.pitch, -35, 35)
            roll_clamped = clamp(self.roll, -25, 25)

            # ---------------- اعمال روی استخوان ----------------
            # اگر دیدی جهت‌ها برعکس‌اند، علامت منفی رو جابه‌جا کن
            self.head_bone.setH(yaw_clamped)  # چپ/راست درست
            self.head_bone.setP(roll_clamped)  # کج شدن طبیعی
            self.head_bone.setR(-pitch_clamped)  # بالا/پایین واقعی

        # پنجرهٔ دیباگ وب‌کم
        cv2.imshow("Webcam Debug", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            # ESC → خروج
            self.cap.release()
            cv2.destroyAllWindows()
            sys.exit()

        return task.cont


# ---------------------------------------------------------
# اجرا
# ---------------------------------------------------------
if __name__ == "__main__":
    app = SimpleHeadTracker()
    app.run()
