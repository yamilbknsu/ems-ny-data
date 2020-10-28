# flake8: noqa
import time
import json
from typing import Optional, List
from abc import ABC, abstractmethod
from heapq import heappush, heappop, heapify

from typing import Optional
import matplotlib.pyplot as plt

def secondsToTimestring(seconds: float) -> str:
    days = int(seconds // 86400)
    seconds = seconds % 86400
    hours = int(seconds // 3600)
    seconds = seconds % 3600
    minutes = int(seconds // 60)
    seconds = seconds % 60

    return 'day {} {:02d}:{:02d}:{:0>5}'.format(days, hours, minutes, '{:.2f}'.format(seconds))

class AbstractEvent(ABC):
    '''
    Abstract Event is a contract between the architect and the programmer, 
    that enforces that any class that inherits from it, must implemente an
    execute method. Additionally, it inherits from ABC (docs.python.org/3/library/abc.html) 
    which is defined as  "a helper class that has ABCMeta as its metaclass. With this class,
    an abstract base class can be created by simply deriving from ABC avoiding sometimes 
    confusing metaclass usage".
    '''
    @abstractmethod
    def execute(self, simulator): 
        pass

class Event(AbstractEvent,ABC):
    '''
    The Event class extends the AbstractEvent. Here we provide a proper __init__ method that
    define the time when the event is created and a proper name is provided to identify the Event
    during tracing.
    '''
    NumberOfEvents = 0 # class variable that keeps track of the number of events created

    def __init__(self, time, name):
        '''
        The Event class constructor that requires two formal arguments.
        
        time:(float) The time when the event must be triggered
        -----
        name:(string) The name of the event, which is useful when we need to combine events from
        -----         different simulation paradigms
        '''
        Event.NumberOfEvents += 1
        self.time = time
        if name is None:
            self.name = "Event " + str(Event.NumberOfEvents)
        else:
            self.name = name 
        self.message = self.name
    
    @abstractmethod
    def execute(self, simulator):
        '''
        This method should be overriden in all the subclasses of Event.

        Returns
        -------
        Optionally, this method may return another event object.
        This is called a chained event. This behaviour is desirable when
        we want to simulate the logic of one event occuring exactly after
        another, without any risk of another happening at the same time interferring.
        
        e.g.
        When an entity arrives, we want to inmediately push it to a queue and check
        if it can continue going forward.
        '''
        pass

    def __lt__(self, y):
        '''
        Overridden __lt__ method to automatically compare by the time variable of the event 
        
        Formal Parameters:
        ------------------
        y:(Event) A proper instance of the Event Class
        
        '''
        if isinstance(y, Event):
            return self.time < y.time
        else:
            raise ValueError('This is not an event')

    def __eq__(self, other):
        '''
        Overridden __eq__ to test for equality between to Event times
        
        Formal Parameters:
        ------------------
        y:(Event) A proper instance of the Event Class
        '''
        return self.__dict__ == other.__dict__
    
    def __str__(self):
        '''
        Implemented __str__ to print time and name of the event (useful for tracing and debugging)
        '''
        return str(self.time)+' , '+self.name


class AbstractSimulator(object):
    '''
    This Class implements the basic structures needed for a more complex Simulator
    '''
    def __init__(self):
        '''
        The constructor of the class
        '''
        self.events = None
        
    def insert(self,e): 
        '''
        Insert an Event object
        
        Formal Parameters:
        ------------------
        e:(Event) An Event to be inserted
        
        Returns:
        --------
        Void
        '''
        self.events.insert(e)
        
    def cancel(self,e): 
        '''
        This method allows the user to cancel an Event
        
        Formal Parameters:
        ------------------
        e:(Event) An Event to be canceled
        
        Returns:
        --------
        Void
        '''
        raise NotImplementedError("Method not implemented")


class Simulator(AbstractSimulator):
    '''
    This class extends the Abstract Simulator by implementing its more relevant methods
    '''
    def __init__(self, metrics = {}, verbose = True):
        '''
        The constructor of the class requires the following parameters.
        
        Formal Parameters:
        ------------------
        metrics:(Dictionary) A dictionary with the metrics to be collected
        verbose:(Boolean) A boolean variable to indicate the verbose level
        '''
        super().__init__()
        self.time = 0
        self.metrics = metrics
        self.verbose = verbose
        self.events = ListQueue() # here you are passing the scheduler

        self.log = []

    def now(self):
        '''
        This methods returns the simulation time
        '''
        return self.time

    def log(self, message):
        '''
        This method is a logger that prints to the screen during execution
        
        Formal Parameters:
        ------------------
        message:(String) It is the message contained in each Event
        
        '''
        print(Simulator.secondsToTimestring(self.now()), message)

    def doAllEvents(self):
        '''
        Here is where the REAL ACTION is happening. The method start setting 
        the simulation time at the very beginning, and then all the events from 
        the queue are removed (if verbose is TRUE, you will see the message each
        Event has inside). Then, the simulation time is advanced to the time of 
        the removed Event and the Event.execute method is called
        '''
        # Clock time at the start of the simulation
        self.start_time = time.time()
        
        # Dequeue all the events
        while self.events.size() > 0:
            e = self.events.removeFirst()
            if self.verbose:
                print(secondsToTimestring(e.time), e.message)
                self.log.append(secondsToTimestring(e.time) + e.message)

            # Update simulation time
            self.time = e.time

            # Execute the event and hold the return value (which might be a chained event)
            chained_event = e.execute(self)

            # If the return value of the event is not None, then it is interpreted
            # as a chained event, and we proceed to execute it. 
            # We do this until no more chained events remain.
            while chained_event is not None:
                if self.verbose:
                    print('{:>17}'.format('### Chained:'), chained_event.message)
                    self.log.append('{:>17}'.format('### Chained:') + chained_event.message)
                chained_event = chained_event.execute(self)

        self.recoverMetrics() 
        return self.metrics
    
    def doOneEvent(self):
        """
        Execute the first event in the queue and
        all of its consecutive chained events

        Returns:
            List[Event]: A list with instances of the executed events
        """
        
        e = self.events.removeFirst()
        if self.verbose:
            print(secondsToTimestring(e.time), e.message)

        self.time = e.time

        # Handling chained events
        chained_event = e.execute(self)
        to_record_events = [e]

        while chained_event is not None:
            to_record_events.append(chained_event)

            if self.verbose:
                print('{:>17}'.format('### Chained:'), chained_event.message)
            chained_event = chained_event.execute(self)

        #for event in to_record_events:
        #    self.recorder.record(event)
        
        return to_record_events
    
    def recoverMetrics(self):
        """
        This method should be overriden with the important metrics
        for the simulation inside the custom simulator model class.
        """
        pass


class ListQueue(object):
    '''
    This class is the one that controls the insertion and deletion of events from the scheduler. 
    We are using the heapq module as scheduler (docs.python.org/2/library/heapq.html).
    '''
    def __init__(self, initEvents: List=[]):
        '''
        This is the class constructor that allows the user to initialize the simulation with 
        some predefined events (by default) 
        
        Fromal Parameters:
        ------------------
        initEvents:(list) It can contain specific events defined by the user. The default values
                        is an empty list (https://docs.python.org/3/library/typing.html)
        '''
        self.elements = initEvents
        heapify(self.elements)  # Initialize the event queue as a heap list

    def insert(self, x):
        '''
        The insert method takes an Event object and place it in the proper location of the scheduler.
        
        Formal Parameters:
        ------------------
        x:(Event) A proper instance of the Event class
        
        Return:(Void)
        -------
        '''
        if isinstance(x, Event):
            heappush(self.elements, x)
            # heapify(self.elements)
        else:
            raise ValueError('This is not an event')

    def removeFirst(self):
        '''
        The removeFirts method pops the first Event object from the scheduler.   
        
        Formal Parameters:
        ------------------
        None
        
        Return:
        -------
        An Event object
        '''
        if len(self.elements) == 0:
            return None
        
        return heappop(self.elements)

    def remove(self, event):
        """
        The move method allows us to remove any event from the scheduler, idependent 
        of its position on it.
        
        *****
        Due to the lack of an implementation of a searching
        algorithm inside the heapq library, the best way to implement
        this is to basically linearly search for the index of the
        desired tuple, pop it from the list, heapify the remaining
        elements, and return the element in question.
        *****
        
        Formal Parameters:
        ------------------
        event:(Event) The Event you want to remove
        
        Return:
        -------
        the removed Event
        """
        # TODO: In the future, we should implement a custom function for the indexed pop of a heap element using the theory of its construction.
        if not isinstance(event, Event):
            raise ValueError('This is not an event')

        # Check if list.index is more efficent than this
        # https://stackoverflow.com/questions/10162679/python-delete-element-from-heap
        # http://www.mathcs.emory.edu/~cheung/Courses/171/Syllabus/9-BinTree/heap-delete.html
        for i in range(len(self.elements)):
            if self.elements[i] == event:
                element = self.elements.pop(i)
                heapify(self.elements)
                return element
        return None

    def empty(self):
        '''
        This method set and empty list for the heap
        '''
        self.elements = list()
        heapify(self.elements)

    def size(self):
        '''
        This method returns the size of heap
        '''
        return len(self.elements)

class SimulationEntity(ABC):
    """
    SimulationEntity [summary]

    Base class for all simulation entities to inherit from.
    You should extend this class to all your objects that are
    going to be used to model entities that flow inside a system.

    :param ABC: Netaclass ABC
    :type ABC: ABC    
    """

    NumberOfEntities = 0

    def __init__(self, name=None):
        """
        __init__ Class constructor

        Optional string value for the name of the entity

        :param name: Entity's name, defaults to None
        :type name: str, optional
        """        
        SimulationEntity.NumberOfEntities += 1

        self.name = name
        if name is None:
            self.name = 'Entity {}'.format(SimulationEntity.NumberOfEntities)
    
    def __str__(self):
        """
        __str__ Modification of the reserved String method

        Return the entity's name

        :return: Entity's name
        :rtype: str
        """        
        return self.name


class Statistic:

    def __init__(self, name, *args):

        self.name = name
        self.data = []

    def record(self, *args):
        pass

    def recordAverage(self, time, value):
        pass
    
    def average(self):
        pass
            
    def max(self):
        pass
    
    def min(self):
        pass


class StateStatistic(Statistic):

    def __init__(self, name, initial_value = 0, initial_time = 0):

        self.name = name
        self.data = [(initial_time, initial_value)]

    def record(self, time, value):
        self.data.append((time, value))

    def recordAverage(self, time, value):
        print("Warning: Formula not working")
        self.data.append((self.average() * self.data[-1][0] + value*(time - self.data[-1][0]))/time)

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
    
    def sum(self):
        return sum(d[1] for d in self.data)

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

class TallyStatistic(Statistic):

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

class TimedTallyStatistic(Statistic):

    def __init__(self, name):

        self.name = name
        self.data = []

    def record(self, time, value):
        self.data.append((time, value))
    
    def recordAverage(self, time, value):
        if len(self.data) > 0:
            self.data.append((time, (self.data[-1][1]*len(self.data) + value)/(len(self.data) + 1)))
        else:
            self.data.append((time, value))
        #self.data.append((time, (self.average()*len(self.data) + value)/(len(self.data)+1)))
    
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


class SpatialStatistic(Statistic):

    def __init__(self, name):

        self.name = name
        self.data = []

    def record(self, time, pos, value):
        self.data.append((time, pos, value))
    
    def recordAverage(self, time, pos, value):
        posData = self.getDataFromPos(pos)
        if len(posData) > 0:
            self.data.append((time, pos, (posData[-1][2]*len(posData) + value)/(len(posData) + 1)))
        else:
            self.data.append((time, pos, value))
    
    def average(self):
        return {p: sum(d[2] for d in self.gefDataFromPos(p))/len(self.gefDataFromPos(p)) if len(self.gefDataFromPos(p)) != 0 else 0 for p in self.getPositions()}

    def getPositions(self):
        return list(set([d[1] for d in self.data]))

    def gefDataFromPos(self, pos):
        return [d for d in self.data if d[1] == pos]
    
    def max(self):
        return max(d[2] for d in self.data)
    
    def min(self):
        return min(d[2] for d in self.data)
    
    def visualize(self, edge_color = '#70cfff', edge_width = 2, fill_color='#1ca6eb', show = True):
        pass

    def __str__(self):
        return 'SpatialStatistic {}: avg: {}, min: {}, max: {}'.format(self.name, 
                                self.average(), self.min(), self.max())

class CounterStatistic(Statistic):

    def __init__(self, name):

        self.name = name
        self.data = 0

    def record(self, repetition = 1):
        self.data += repetition
    
    def value(self):
        return self.data
    
    def __str__(self):
        return 'CounterStatistic {}: value: {}'.format(self.name, self.value())
