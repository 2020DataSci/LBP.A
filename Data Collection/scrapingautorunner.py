import time
import os
import threading
import logging
import msvcrt


def scraping():
    os.system('python .\scrapingscript.py')

running = True
while running:
    t = threading.Thread(target = scraping)
    t.start()
    t.join()
    
    if msvcrt.kbhit():
            running = False