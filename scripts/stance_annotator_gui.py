import random
import sys

import pandas as pd

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class StanceAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("כלי תיוג עמדות ונושאים (Deduplicated) - Annotator B")
        self.setGeometry(100, 100, 1000, 800)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #f5f5f5; color: #000; font-family: 'Arial'; }
            QTextEdit { background-color: #fff; color: #000; border: 2px solid #aaa; font-size: 20px; }
            QLineEdit { background-color: #fff; color: #000; border: 1px solid #999; padding: 8px; font-size: 16px; }
            QRadioButton { font-size: 17px; font-weight: bold; }
            QPushButton { font-weight: bold; padding: 10px; border-radius: 4px; border: 1px solid #999; }
        """)

        self.df = None
        self.history = []
        self.current_index = -1
        self.current_side = ""
        self.current_text = ""
        self.file_path = ""
        self.prefix = "annotator_B_"
        self.total_unique_tasks = 0

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        top_layout = QHBoxLayout()
        self.btn_load = QPushButton("📂 טען CSV")
        self.btn_load.clicked.connect(self.load_file)
        self.btn_save = QPushButton("💾 שמור קובץ...")
        self.btn_save.clicked.connect(self.save_file)
        top_layout.addWidget(self.btn_load)
        top_layout.addWidget(self.btn_save)
        main_layout.addLayout(top_layout)

        self.status_frame = QFrame()
        self.status_frame.setStyleSheet("background-color: #333; color: white; border-radius: 0px;")
        status_layout = QHBoxLayout(self.status_frame)
        self.lbl_task_type = QLabel("ממתין לטעינה...")
        self.lbl_counter = QLabel("משימות: 0 / 0")
        self.lbl_info = QLabel("-")
        for lbl in [self.lbl_task_type, self.lbl_counter, self.lbl_info]:
            lbl.setStyleSheet("color: white; font-weight: bold; font-size: 16px;")
            status_layout.addWidget(lbl)
        main_layout.addWidget(self.status_frame)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        main_layout.addWidget(self.text_display)

        input_frame = QFrame()
        input_frame.setStyleSheet("background-color: #e0e0e0; padding: 15px; border-top: 2px solid #bbb;")
        form_layout = QFormLayout(input_frame)
        self.input_topic = QLineEdit()
        self.input_topic.setPlaceholderText("הזן נושא...")
        self.input_topic.returnPressed.connect(self.submit_data)
        form_layout.addRow(QLabel("<b>נושא:</b>"), self.input_topic)

        stance_widget = QWidget()
        stance_layout = QHBoxLayout(stance_widget)
        self.radio_group = QButtonGroup(self)
        self.rb_neutral = QRadioButton("נייטרלי (1)")
        self.rb_favor = QRadioButton("בעד (2)")
        self.rb_against = QRadioButton("נגד (3)")
        for rb in [self.rb_neutral, self.rb_favor, self.rb_against]:
            self.radio_group.addButton(rb)
            stance_layout.addWidget(rb)
        form_layout.addRow(QLabel("<b>עמדה:</b>"), stance_widget)
        main_layout.addWidget(input_frame)

        btn_layout = QHBoxLayout()
        self.btn_back = QPushButton("⬅️ חזור אחורה")
        self.btn_back.setStyleSheet("background-color: #ff9800; color: white; font-size: 16px;")
        self.btn_back.clicked.connect(self.undo_step)

        self.btn_submit = QPushButton("✅ שמור והמשך (Enter)")
        self.btn_submit.setStyleSheet("background-color: #2e7d32; color: white; font-size: 18px;")
        self.btn_submit.clicked.connect(self.submit_data)

        self.btn_skip = QPushButton("⏭️ דלג (Delete)")
        self.btn_skip.clicked.connect(self.next_random)

        btn_layout.addWidget(self.btn_back, 1)
        btn_layout.addWidget(self.btn_submit, 3)
        btn_layout.addWidget(self.btn_skip, 1)
        main_layout.addLayout(btn_layout)

    def key_press_event(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_1:
            self.rb_neutral.setChecked(True)
        elif event.key() == Qt.Key.Key_2:
            self.rb_favor.setChecked(True)
        elif event.key() == Qt.Key.Key_3:
            self.rb_against.setChecked(True)
        elif event.key() in [Qt.Key.Key_Return, Qt.Key.Key_Enter]:
            self.submit_data()
        elif event.key() == Qt.Key.Key_Delete:
            self.next_random()
        else:
            super().keyPressEvent(event)

    def is_empty(self, val):
        if pd.isna(val):
            return True
        s = str(val).strip().lower()
        return s in ["", "nan", "none"]

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "טען CSV", "", "CSV Files (*.csv)")
        if path:
            self.df = pd.read_csv(path)
            self.file_path = path
            for side in ["summary", "article"]:
                for suffix in ["topic", "stance"]:
                    col = f"{self.prefix}{side}_{suffix}"
                    if col not in self.df.columns:
                        self.df[col] = ""
                    self.df[col] = self.df[col].astype(object)

            # Initial Total Count
            s_texts = self.df["Sentence in Summary"].dropna().unique().tolist()

            def get_art_text(row):
                m = row.get("Manual match")
                return m if not self.is_empty(m) else row.get("Best Match Sentences From Article", "")

            a_texts = self.df.apply(get_art_text, axis=1).unique().tolist()
            self.total_unique_tasks = len([t for t in s_texts if not self.is_empty(t)]) + len(
                [t for t in a_texts if not self.is_empty(t)]
            )
            self.next_random()

    def get_remaining_tasks(self):
        s_col_text = "Sentence in Summary"
        s_col_topic = f"{self.prefix}summary_topic"
        untagged_summary_df = self.df[self.df[s_col_topic].apply(self.is_empty)]
        unique_summary_texts = untagged_summary_df[s_col_text].unique().tolist()

        def get_art_text(row):
            m = row.get("Manual match")
            return m if not self.is_empty(m) else row.get("Best Match Sentences From Article", "")

        a_col_topic = f"{self.prefix}article_topic"
        untagged_article_df = self.df[self.df[a_col_topic].apply(self.is_empty)].copy()
        unique_article_texts = (
            untagged_article_df.apply(get_art_text, axis=1).unique().tolist() if not untagged_article_df.empty else []
        )

        pool = [(t, "summary") for t in unique_summary_texts if not self.is_empty(t)] + [
            (t, "article") for t in unique_article_texts if not self.is_empty(t)
        ]
        return pool

    def next_random(self):
        if self.df is None:
            return
        pool = self.get_remaining_tasks()
        if pool:
            if self.current_text:
                self.history.append((self.current_text, self.current_side))
            self.current_text, self.current_side = random.choice(pool)
            self.display_current()
        else:
            QMessageBox.information(self, "סיום", "כל המשימות הושלמו!")

    def undo_step(self):
        if not self.history:
            QMessageBox.warning(self, "אופס", "אין צעדים נוספים לחזור אליהם!")
            return
        self.current_text, self.current_side = self.history.pop()
        self.display_current()

    def update_status(self):
        if self.df is None:
            return
        pool = self.get_remaining_tasks()
        # FIX: Explicitly count unique tasks remaining
        remaining = len(pool)
        self.lbl_counter.setText(f"משימות ייחודיות שנותרו: {remaining} / {self.total_unique_tasks}")
        self.lbl_task_type.setText(f"מקור: {'סיכום' if self.current_side == 'summary' else 'מאמר'}")

        if self.current_side == "summary":
            count = len(self.df[self.df["Sentence in Summary"] == self.current_text])
        else:

            def is_match(row):
                m = row.get("Manual match")
                best = row.get("Best Match Sentences From Article")
                return (m if not self.is_empty(m) else best) == self.current_text

            count = self.df.apply(is_match, axis=1).sum()
        self.lbl_info.setText(f"מעדכן {count} שורות")

    def display_current(self):
        title = "משפט מהסיכום" if self.current_side == "summary" else "משפט מהמאמר"
        self.text_display.setHtml(
            f"<div style='background-color:white; padding:10px;'><h2>{title}:</h2>"
            "<p style='font-size:22px; color:#333;'>{self.current_text}</p></div>"
        )
        self.input_topic.clear()
        self.radio_group.setExclusive(False)
        for rb in self.radio_group.buttons():
            rb.setChecked(False)
        self.radio_group.setExclusive(True)
        self.update_status()
        self.input_topic.setFocus()

    def submit_data(self):
        if self.df is None or not self.current_text:
            return
        topic = self.input_topic.text().strip()
        if not topic:
            return

        selected = self.radio_group.checkedButton()
        stance = selected.text().split(" ")[0] if selected else ""

        col_t = f"{self.prefix}{self.current_side}_topic"
        col_s = f"{self.prefix}{self.current_side}_stance"

        if self.current_side == "summary":
            self.df.loc[self.df["Sentence in Summary"] == self.current_text, [col_t, col_s]] = [topic, stance]
        else:

            def is_match(row):
                m = row.get("Manual match")
                best = row.get("Best Match Sentences From Article")
                return (m if not self.is_empty(m) else best) == self.current_text

            mask = self.df.apply(is_match, axis=1)
            self.df.loc[mask, [col_t, col_s]] = [topic, stance]

        self.next_random()

    def save_file(self):
        if self.df is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "שמור קובץ", self.file_path, "CSV Files (*.csv)")
        if path:
            self.df.to_csv(path, index=False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = StanceAnnotator()
    w.show()
    sys.exit(app.exec())
