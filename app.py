from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import uuid

from services.text_checker import check_text
from services.image_checker import check_image
from services.video_checker import check_video_authenticity

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "uploads"
ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


def allowed_file(filename, file_type):
    ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

    if file_type == "image":
        return ext in ALLOWED_IMAGE_EXTENSIONS
    if file_type == "video":
        return ext in ALLOWED_VIDEO_EXTENSIONS
    return False


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        content_type = request.form.get("type")

        
        if content_type == "text":
            text = request.form.get("text", "").strip()

            if not text:
                flash("Please enter some text.")
                return redirect(url_for("index"))

            result = check_text(text)
            return render_template("result.html", result=result)

        
        if content_type in ["image", "video"]:
            if "file" not in request.files:
                flash("No file uploaded.")
                return redirect(url_for("index"))

            file = request.files["file"]

            if file.filename == "":
                flash("No file selected.")
                return redirect(url_for("index"))

            if not allowed_file(file.filename, content_type):
                flash("Invalid file type.")
                return redirect(url_for("index"))

            
            filename = secure_filename(file.filename)
            unique_name = str(uuid.uuid4()) + "_" + filename
            path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(path)

           
            if content_type == "image":
                result = check_image(path)
            else:
                result = check_video_authenticity(path)

            return render_template("result.html", result=result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)