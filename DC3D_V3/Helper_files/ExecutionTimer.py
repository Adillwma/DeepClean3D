import time
import matplotlib.pyplot as plt
import numpy as np
from sys import getsizeof as sys_getsizeof

class Execution_Timer():
    def __init__(self):
        self.reset()

    def record_time(self, event_name="", event_type=""):
        if event_type == "start":
            # Check if event name is already a dictiomnary key in 
            if event_name not in self.data:
                self.data[event_name] = [["start", time.perf_counter() - self.t0]]
            else:
                self.data[event_name].append(["start", time.perf_counter() - self.t0])

        elif event_type == "stop":
            if event_name not in self.data:
                raise ValueError(f"Event: {event_name}, was not started, so cannot be stopped")
            else:
                self.data[event_name].append(["stop", time.perf_counter() - self.t0])
    
        else:
            print("Event type not recognized")

    def return_data(self):
        return self.data
    
    def return_memory_usage(self):
        # calulate memory usage of the self.data dictionary
        memory_usage = sys_getsizeof(self.data)
        return f"Memory usage of the Execution_Timer: {memory_usage} bytes"

    def return_plot(self, cmap='gist_rainbow', dark_mode=False):
        y_centers = np.arange(len(self.data))  # Center y positions for each class

        fig, ax = plt.subplots()

        # create colourmap for each event
        cmap = plt.cm.get_cmap(cmap, len(self.data))

        for occurence_index, (event_name, event_data) in enumerate(self.data.items()):
            num_tuples = len(event_data)

            # create two lists one for start times and one for stop times based on the event data string in the tuple either "start" or "stop"
            start_times = [event_data[i][1] for i in range(num_tuples) if event_data[i][0] == "start"]
            stop_times = [event_data[i][1] for i in range(num_tuples) if event_data[i][0] == "stop"]

            assert len(start_times) == len(stop_times), f"Event: {event_name}, does not have a matching start and stop.\n Please make sure you have a start and stop for each ExecutionTimer event, and that they share the same event name (check for typos)."

            for start_time, stop_time in zip(start_times, stop_times):
                ax.fill_betweenx([y_centers[occurence_index]-0.5, y_centers[occurence_index]+0.5], start_time, stop_time, alpha=0.8, color=cmap(occurence_index))


        # Set y-labels using class names
        ax.set_yticks(y_centers)
        ax.set_yticklabels(list(self.data.keys()))

        ax.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5, axis='x')
        ax.set_xlabel('Time') # Units? auto scale?
        #ax.set_ylabel('Event')
        ax.set_title('Event Timeline')

        if dark_mode:
            # set entire figure and plot background to black and the text and axis to white
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')
        return fig

    def return_times(self):
        for occurence_index, (event_name, event_data) in enumerate(self.data.items()):
            num_tuples = len(event_data)

            # create two lists one for start times and one for stop times based on the event data string in the tuple either "start" or "stop"
            start_times = [event_data[i][1] for i in range(num_tuples) if event_data[i][0] == "start"]
            stop_times = [event_data[i][1] for i in range(num_tuples) if event_data[i][0] == "stop"]

            assert len(start_times) == len(stop_times), f"Event: {event_name}, does not have a matching start and stop.\n Please make sure you have a start and stop for each ExecutionTimer event, and that they share the same event name (check for typos)."

            cummulative_time = 0
            for call_idx, (start_time, stop_time) in enumerate(zip(start_times, stop_times)):
                #print(f"Event: {event_name}, Call: {call_idx+1}, Duration: {stop_time-start_time}")
                cummulative_time += stop_time - start_time
            avg_time = cummulative_time / len(start_times)
            print(f"Event: {event_name}, Average time: {avg_time}, Total time: {cummulative_time}")

    def reset(self):
        self.data = {}
        self.t0 = time.perf_counter()         # set to current time 



if __name__ == "__main__":
    execution_timer = Execution_Timer()

    execution_timer.record_time(event_name="Search for Higgs", event_type="start")
    time.sleep(1)
    execution_timer.record_time(event_name="Search for Higgs", event_type="stop")

    time.sleep(1)


    execution_timer.record_time(event_name="Search for Dark Matter", event_type="start")
    time.sleep(2)
    execution_timer.record_time(event_name="Search for Dark Matter", event_type="stop")

    execution_timer.record_time(event_name="Search for Higgs", event_type="start")
    time.sleep(1)
    execution_timer.record_time(event_name="Search for Higgs", event_type="stop")
    """
    execution_timer.record_time(event_name="Search for Dark Matter", event_type="start")
    time.sleep(2)
    execution_timer.record_time(event_name="Search for Dark Matter", event_type="stop")

    execution_timer.record_time(event_name="Crytalisation", event_type="start")
    time.sleep(2)
    execution_timer.record_time(event_name="Crytalisation", event_type="stop")

    execution_timer.record_time(event_name="Search for Dark Matter", event_type="start")
    time.sleep(1)
    execution_timer.record_time(event_name="Search for Dark Matter", event_type="stop")

    execution_timer.record_time(event_name="Search for Higgs", event_type="start")
    time.sleep(1)
    execution_timer.record_time(event_name="Search for Higgs", event_type="stop")

    execution_timer.record_time(event_name="Crytalisation", event_type="start")
    time.sleep(2)
    execution_timer.record_time(event_name="Crytalisation", event_type="stop")

    execution_timer.record_time(event_name="Search for Higgs", event_type="start")
    time.sleep(1)
    execution_timer.record_time(event_name="Search for Higgs", event_type="stop")

    execution_timer.record_time(event_name="Search for Dark Matter", event_type="start")
    time.sleep(1)
    execution_timer.record_time(event_name="Search for Dark Matter", event_type="stop")

    """
    fig = execution_timer.return_plot(dark_mode=True)
    plt.show()

    execution_timer.return_memory_usage()
    data = execution_timer.return_data()
    #print(data)


