from apscheduler.schedulers.background import BackgroundScheduler
from services import whatsapp_messages
scheduler = BackgroundScheduler()

def start_scheduler():
    if not scheduler.running:
        print("Scheduler function entered ")

        scheduler.add_job(
            lambda:whatsapp_messages("YOU ARE HACKED DADDY !!"),
            trigger="interval",
            # hour=23,
            # minutes=2
            seconds=10
        )

        scheduler.start()
        print("Scheduler started successfully ")
