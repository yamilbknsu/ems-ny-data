# flake8: noqa
import time
import matplotlib.pyplot as plt

from typing import Optional

def secondsToTimestring(seconds: float) -> str:
    days = int(seconds // 86400)
    seconds = seconds % 86400
    hours = int(seconds // 3600)
    seconds = seconds % 3600
    minutes = int(seconds // 60)
    seconds = seconds % 60

    return 'day {} {:02d}:{:02d}:{:0>5}'.format(days, hours, minutes, '{:.2f}'.format(seconds))

class AbstractSimulator(object):

    def __init__(self):
        self.events = None

    def insert(self, e):  # Insert an abstract event
        self.events.insert(e)

    def cancel(self, e):  # AbstractEvent
        raise NotImplementedError("Method not implemented")


class Event:

    N_events = 0

    def __init__(self, time, name):
        self.time = time

        Event.N_events += 1
        if name is None:
            self.name = "Event #" + str(Event.N_events)
        else:
            self.event = name

    def __lt__(self, y):
        if isinstance(y, Event):
            return self.time < y.time
        else:
            raise ValueError('This is not an event')

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class Simulator(AbstractSimulator):

    def __init__(self, metrics = {}, verbose = True):
        super().__init__()
        self.time = 0
        self.metrics = metrics
        self.verbose = verbose

        self.events = ListQueue()

    def now(self):
        return self.time

    def do_all_events(self):
        self.start_time = time.time()
        
        while self.events.size() > 0:
            e = self.events.remove_first()
            if self.verbose:
                print(secondsToTimestring(e.time), e.message)

            self.time = e.time

            # Handling chained events
            chained_event = e.execute(self)
            while chained_event is not None:
                if self.verbose:
                    print('{:>17}'.format('### Chained:'), chained_event.message)
                chained_event = chained_event.execute(self)

        self.recoverMetrics() 
        return self.metrics
    
    def recoverMetrics(self):
        pass


class ListQueue:
    
    elements: list = list()

    def insert(self, x):
        i = 0
        while i < len(self.elements) and self.elements[i] < x:
            i += 1
        self.elements.insert(i, x)

    def remove_first(self):
        if len(self.elements) == 0:
            return None
        x = self.elements.pop(0)
        return x

    def remove(self, x):
        for i in range(len(self.elements)):
            if self.elements[i] == x:
                return self.elements.pop(i)
        return None

    def empty(self):
        self.elements = list()

    def size(self):
        return len(self.elements)

class SimulationEntity:

    N_ENTITIES = 0

    def __init__(self, name: Optional[str]):
        SimulationEntity.N_ENTITIES += 1

        self.name = name
        if name is None:
            self.name = 'Entity {}'.format(SimulationEntity.N_ENTITIES)
    
    def __str__(self):
        return self.name

class StateStatistic:

    def __init__(self, name, initial_value = 0, initial_time = 0):

        self.name = name
        self.data = [(initial_time, initial_value)]

    def record(self, time, value):
        self.data.append((time, value))
    
    def average(self):
        avg = 0
        count = len(self.data)
        for d in range(count - 1):
            avg += self.data[d][1]*(self.data[d+1][0] - self.data[d][0])/self.data[count-1][0]
        return avg
    
    def max(self):
        return max(d[1] for d in self.data)
    
    def min(self):
        return min(d[1] for d in self.data)

    def visualize(self, show_mean = True):
        x = []
        y = []
        for i in range(len(self.data) - 1):
            if self.data[i][0] != self.data[i+1][0]:
                x.append(self.data[i][0])
                x.append(self.data[i+1][0])
                y.append(self.data[i][1])
                y.append(self.data[i][1])
        plt.plot(x, y, '-')
        
        if show_mean:
            plt.plot([0, self.data[-1][0]], [self.average()]*2, 'r-')
        
        plt.title('StateStatistic ' + self.name + ' over time')
        plt.show()
    
    def __str__(self):
        return 'StateStatistic {}: avg: {}, min: {}, max: {}'.format(self.name, 
                                self.average(), self.min(), self.max())

class TallyStatistic:

    def __init__(self, name):

        self.name = name
        self.data = []

    def record(self, value):
        self.data.append(value)
    
    def average(self):
        return sum(self.data)/len(self.data) if len(self.data) != 0 else 0
    
    def max(self):
        return max(d for d in self.data)
    
    def min(self):
        return min(d for d in self.data)

    def __str__(self):
        return 'TallyStatistic {}: avg: {}, min: {}, max: {}'.format(self.name, 
                                self.average(), self.min(), self.max())

class TimedTallyStatistic:

    def __init__(self, name):

        self.name = name
        self.data = []

    def record(self, time, value):
        self.data.append((time, value))
    
    def average(self):
        return sum(d[1] for d in self.data)/len(self.data) if len(self.data) != 0 else 0
    
    def max(self):
        return max(d[1] for d in self.data)
    
    def min(self):
        return min(d[1] for d in self.data)
    
    def visualize(self, edge_color = '#70cfff', edge_width = 2, fill_color='#1ca6eb', show = True):
        for d in self.data:
            plt.plot(d[0], d[1], 'o', markeredgewidth=edge_width, markeredgecolor = edge_color, markerfacecolor=fill_color)

        if show:
            plt.title('TimedTallyStatistic ' + self.name + ' over time')
            plt.show()

    def __str__(self):
        return 'TimedTallyStatistic {}: avg: {}, min: {}, max: {}'.format(self.name, 
                                self.average(), self.min(), self.max())

class CounterStatistic:

    def __init__(self, name):

        self.name = name
        self.data = 0

    def record(self, repetition = 1):
        self.data += repetition
    
    def value(self):
        return self.data
    
    def __str__(self):
        return 'CounterStatistic {}: value: {}'.format(self.name, self.value())
    

class EventRecorder:

    def __init__(self):
        self.executed_events = []
    
    def record(self, event):
        self.executed_events.append(event)
    
    def getEvents(self):
        output_events = [{**{key:str(value) for key, value in e.__dict__.items()}, **{'type':type(e).__name__}} for e in self.executed_events]
        self.executed_events = []
        return output_events