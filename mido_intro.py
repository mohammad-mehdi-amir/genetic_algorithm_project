import os
from mido import Message, MidiFile, MidiTrack

def save_midi(sequence, filename, tempo=100):
   
    output_folder = "melody"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    
 
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    

    track.append(Message('program_change', program=46))  
    

    for note in sequence:
        note = max(0, min(note, 127))
        
        track.append(Message('note_on', note=note, velocity=64, time=tempo))

        track.append(Message('note_off', note=note, velocity=64, time=tempo))
    

    file_path = os.path.join(output_folder, f'generation_{filename[0]}-fitness_{round(filename[1])}.mid')
    

    mid.save(file_path)
    
    


def clear_melody_folder(folder_name="melody"):

    folder_path = os.path.join(os.getcwd(), folder_name)
    
    if os.path.exists(folder_path):  
        for file in os.listdir(folder_path): 
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path): 
                os.remove(file_path)  
        
    else:
        print(f"The folder '{folder_name}' does not exist.")
        
