'''
Liste d'évènements KH
 - date de début au format YYYY-MM-DD HH:mm
 - date de fin   au format YYYY-MM-DD HH:mm
'''

class Event:
    def __init__(self, start, end):
        self.start = start
        self.end = end

events = [
    Event("2009-09-29 21:27", "2009-09-29 21:29"),
    Event("2011-05-15 09:13", "2011-05-15 09:30"),
    Event("2011-06-11 12:26", "2011-06-11 12:37"),
    Event("2011-06-12 00:28", "2011-06-12 00:36"),
    Event("2011-06-15 13:10", "2011-06-15 13:20"),
    Event("2011-06-17 00:45", "2011-06-17 00:58"),
    Event("2011-06-19 23:33", "2011-06-19 23:43"),
]