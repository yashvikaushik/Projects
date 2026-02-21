from apscheduler.schedulers.background import BackgroundScheduler
from services import whatsapp_messages
scheduler = BackgroundScheduler()

def start_scheduler():
    if not scheduler.running:
        print("Scheduler function entered ")

        scheduler.add_job(
            lambda:whatsapp_messages("YOU ARE HACKED DADDY !!"),
            trigger="cron",
            hour=23,
            minute=45
        )

        scheduler.start()
        print("Scheduler started successfully ")
