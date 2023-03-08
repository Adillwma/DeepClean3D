# -*- coding: utf-8 -*-
"""
System Information Check V1.0.0
Author: Adill Al-Ashgar
Created on Fri March 3 2023

#Return all vital computer statsistics to the main code body for adding to network summary, developed as helper file due to length of code (Trying to keep main file readable and clean)
"""

import psutil
import platform
from datetime import datetime
cuda = False



def get_system_information():

    def get_size(bytes, suffix="B"):
        """
        Scale bytes to its proper format
        """
        factor = 1024
        for unit in ["", "K", "M", "G", "T", "P"]:
            if bytes < factor:
                return f"{bytes:.2f}{unit}{suffix}"
            bytes /= factor

    def gpu_info():
        """
        For more detailed GPU info, added in try except stament as need pip install for wmi and trying to limit amount of dependencies that need install from breaking the main programs running
        """
        try:
            import wmi
            GPU_devices = wmi.WMI().Win32_VideoController()

            gpu_data = list()

            for GPU_id, GPU_device in enumerate(GPU_devices):
                controller_info = {
                    f'GPU {GPU_id+1}': GPU_device.wmi_property('Name').value,
                    f'GPU {GPU_id+1} VRAM': get_size(GPU_device.wmi_property('AdapterRAM').value),
                }
            gpu_data.append(controller_info)

        except:
            print("GPU Identification failed due to missing 'wmi' Python lib, for more GPU info please 'pip install wmi'!")
            controller_info = {
                    'Name': "Identification failed due to missing 'wmi' Python lib please pip install wmi",
                    'VRAM': "Identification failed due to missing 'wmi' Python lib please pip install wmi",
                }
            gpu_data.append(controller_info)

        return (gpu_data)

    # Create an empty string to store the output
    system_information = ""

    # Add Host OS information to the system_info string
    system_information += "="*40 + " Host OS Information " + "="*40 + "\n"
    uname = platform.uname()
    system_information += f"System: {uname.system}\n"
    system_information += f"Release: {uname.release}\n"
    system_information += f"Version: {uname.version}\n"
    system_information += f"Machine: {uname.machine}\n"

    # Add CPU information to the system_information string
    system_information += "="*40 + " CPU Info " + "="*40 + "\n"
    # number of cores
    system_information += f"Processor: {uname.processor}\n"
    system_information += "Physical cores: " + str(psutil.cpu_count(logical=False)) + "\n"
    system_information += "Total cores: " + str(psutil.cpu_count(logical=True)) + "\n"

    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    system_information += f"Max Frequency: {cpufreq.max:.2f}Mhz\n"
    system_information += f"Min Frequency: {cpufreq.min:.2f}Mhz\n"

    # Add GPU information to the system_information string
    system_information += "="*40 + " GPU Info " + "="*40 + "\n"
    # CUDA GPU found?
    system_information += "CUDA GPU Available: " + str(cuda) + "\n"
    GPU_enumeration = gpu_info()
    for GPU_dev in GPU_enumeration:
        for key, value in GPU_dev.items():
            system_information += f"{key}: {value}\n"

    # Add Memory Information to the system_information string
    system_information += "="*40 + " Memory Information " + "=" *40 + "\n"
    # get the memory details
    svmem = psutil.virtual_memory()
    system_information += "="*10 + "RAM" + "=" * 10 + "\n"
    system_information += f"Total: {get_size(svmem.total)}\n"
    system_information += f"Available: {get_size(svmem.available)}\n"
    system_information += f"Used: {get_size(svmem.used)}\n"
    system_information += f"Percentage: {svmem.percent}%\n"

    system_information += "="*10 + "Virtual Memory" + "=" * 10 + "\n"
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    system_information += f"Total: {get_size(swap.total)}\n"
    system_information += f"Free: {get_size(swap.free)}\n"
    system_information += f"Used: {get_size(swap.used)}\n"
    system_information += f"Percentage: {swap.percent}%\n"
    system_information += "\n"

    return system_information