# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import cProfile




# %%
c=299792458 #m/s speed of light



# %% [markdown]
# synthetic amorphous fused silica 
# 
#  “Fused quartz” and “fused silica” are medium refractive-index glasses containing predominately SiO2 in the amorphous (non-crystalline) form. The word quartz usually refers to the natural crystal or mineral as opposed to the phrase “fused quartz” which refers to the glass that is created out of a manufacturing process which involves heating quartz crystals to temperatures of around 2000 degrees Celsius (which has lower refractive index).
# 
# Fused quartz has better ultraviolet transmission than most other common glasses (such as borosilicate glasses that have somewhat higher refractive indices), making it an ideal candidate for applications in the sub-400nm spectral region. Additionally fused quartz/fused silica have a low thermal expansion coefficient

# %% [markdown]
# The colliding-beam LHCb experiment has a forward spectrometer geometry covering ±10-300 mrad horizontally and ±10-250 mrad vertically relative to the beam axis

# %% [markdown]
# The timing information is made up of two parts, ToF & ToP. The first part is the particle time-of-flight (ToF) over a
# flight path of 10 m from the LHCb interaction point (IP) to
# the radiator plate. The second part is the Cherenkov photons’
# time-of-propagation (ToP) in the radiator plate. The photon arrival
# time is measured by micro-channel plate (MCP) sensors which is a combination of the ToF and the ToP.

# %%
def set_label(ax):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') 

class EventCounter:
    def __init__(self, num_of_particles, n_cher_per_track):
        self.num_tracks_hit_radiator = 0
        self.num_photons_left_radiator = 0
        self.num_photons_left_mouth = 0
        self.num_photons_hit_mirror = 0
        self.num_photons_hit_mcp = 0
        self.num_of_particles = num_of_particles
        self.n_cher_per_track = n_cher_per_track

    def add_track(self):
        self.num_tracks_hit_radiator += 1

    def count_photons(self, photon):
        if photon.left_radiator:
            self.num_photons_left_radiator += 1
            if photon.left_mouth:
                self.num_photons_left_mouth += 1
                if photon.hit_mirror:
                    self.num_photons_hit_mirror += 1
                    if photon.hit_mcp:
                        self.num_photons_hit_mcp += 1

    def print_info(self):
        num_of_missed_tracks = self.num_of_particles - self.num_tracks_hit_radiator
        num_of_photons = self.num_tracks_hit_radiator * self.n_cher_per_track

        print("DATA CHECK MAIN")
        print(f"Number of tracks genrated: {self.num_of_particles}")
        print(f"Number of tracks hitting radiator: {self.num_tracks_hit_radiator}")
        print(f"Number of tracks missing radiator: {num_of_missed_tracks}")
        print(f"Number of expected photons on mcp: {num_of_photons}")
        print(f"Number of actual photons on mcp: {self.num_photons_hit_mcp}")

        if self.num_tracks_hit_radiator != 0:
            print(f"Percentage of photons reaching mcp: {self.num_photons_hit_mcp / num_of_photons * 100}%\n")

            print("DATA CHECK QUARTZ")
            print(f"Number of photons leaving quartz: {self.num_photons_left_radiator}")
            print(f"Percentage of photons leaving quartz: {self.num_photons_left_radiator / num_of_photons * 100}%\n")

            print("DATA CHECK FOCUSING BLOCK")
            print(f"Number of photons leaving focusing block: {self.num_photons_left_mouth}")
            print(f"Percentage of photons leaving focusing block: {self.num_photons_left_mouth / num_of_photons * 100}%\n")

            print("DATA CHECK MIRROR")
            print(f"Number of photons reaching mirror: {self.num_photons_hit_mirror}")
            print(f"Percentage of photons reaching mirror: {self.num_photons_hit_mirror / num_of_photons * 100}%\n")

            print("DATA CHECK MCP")
            print(f"Number of photons reaching MCP: {self.num_photons_hit_mcp}")
            print(f"Percentage of photons reaching MCP: {self.num_photons_hit_mcp / num_of_photons * 100}%\n")


# %%
# Visualisation functions
def define_quartz(quartz_params, quartz_position):    ##### SETS UP THE QUARTZ RADIATOR WITH THE SPECIFIED POSITION USED AS THE CENTRE OF THE FRONT FACE
    quartz_depth, quartz_height, quartz_width = quartz_params
    qax, qay, qaz = quartz_position
    
    half_quartz_width = quartz_width / 2
    half_quartz_height = quartz_height / 2
    
    # Define vertices with symmetry on y and z dimensions
    Z = np.array([
        [qax, qay - half_quartz_width, qaz - half_quartz_height],
        [qax, qay - half_quartz_width, qaz + half_quartz_height + 0.03],
        [qax, qay + half_quartz_width, qaz + half_quartz_height + 0.03],
        [qax, qay + half_quartz_width, qaz - half_quartz_height],
        [qax + quartz_depth, qay - half_quartz_width, qaz - half_quartz_height],
        [qax + quartz_depth, qay - half_quartz_width, qaz + half_quartz_height],
        [qax + quartz_depth, qay + half_quartz_width, qaz + half_quartz_height],
        [qax + quartz_depth, qay + half_quartz_width, qaz - half_quartz_height]
    ])
    
    verts = [
        [Z[0], Z[1], Z[2], Z[3]],
        [Z[4], Z[5], Z[6], Z[7]],
        [Z[0], Z[1], Z[5], Z[4]],
        [Z[2], Z[3], Z[7], Z[6]],
        [Z[1], Z[2], Z[6], Z[5]],
        [Z[4], Z[7], Z[3], Z[0]]
    ]

    return verts

def set_detector_object_positions(quartz_params, quartz_position):
    qax, qay, qaz = quartz_position
    quartz_depth, quartz_height, quartz_width = quartz_params
    half_quartz_width = quartz_width / 2
    half_quartz_height = quartz_height / 2

    # BACK WALL
    back_coords = np.array([[qax, qay - half_quartz_width, qaz + half_quartz_height],# Bottom-left corner
                            [qax, qay + half_quartz_width, qaz + half_quartz_height], # Bottom-right corner
                            [qax, qay + half_quartz_width, qaz + half_quartz_height + 0.03], # Top-right corner + 3cm comes from the images in forty
                            [qax, qay - half_quartz_width, qaz + half_quartz_height + 0.03],# Top-left corner + 3cm comes from the images in forty
                            [qax, qay - half_quartz_width, qaz + half_quartz_height] # Bottom-left corner home
                            ])

    ### MOUTH OF FOCUSING BLOCK ###
    mouth_coords = np.array([[qax + quartz_depth, qay - half_quartz_width, qaz + half_quartz_height],# Bottom-left corner
                            [qax+ quartz_depth, qay + half_quartz_width, qaz + half_quartz_height], # Bottom-right corner
                            [qax, qay + half_quartz_width, qaz + half_quartz_height + 0.03], # Top-right corner + 3cm comes from the images in forty
                            [qax, qay - half_quartz_width, qaz + half_quartz_height + 0.03],# Top-left corner + 3cm comes from the images in forty
                            [qax+ quartz_depth, qay - half_quartz_width, qaz + half_quartz_height] # Bottom-left corner home
                            ])        

    # Define the corner coordinates
    mirror_coords = np.array([[qax + 0.12, qay - half_quartz_width, qaz + half_quartz_height + 0],  # Bottom-left corner   # size and position of mirror taken from forty 
                            [qax + 0.12, qay + half_quartz_width, qaz + half_quartz_height + 0],  # Bottom-right corner
                            [qax + 0.10, qay + half_quartz_width, qaz + half_quartz_height + 0.14],  # Top-right corner
                            [qax + 0.10, qay - half_quartz_width, qaz + half_quartz_height + 0.14],   # Top-left corner
                            [qax + 0.12, qay - half_quartz_width, qaz + half_quartz_height + 0]  # Bottom-left corner   # size and position of mirror taken from forty 
                            ])

    # Define the corner coordinates
    mcp_coords = np.array([[qax, qay - half_quartz_width, qaz + half_quartz_height + 0.03],# Bottom-left corner
                           [qax, qay + half_quartz_width, qaz + half_quartz_height + 0.03], # Bottom-right corner
                           [qax-0.04, qay + half_quartz_width, qaz + half_quartz_height + 0.08], # Top-right corner + 3cm comes from the images in forty
                           [qax-0.04, qay - half_quartz_width, qaz + half_quartz_height + 0.08],# Top-left corner + 3cm comes from the images in forty
                           [qax, qay - half_quartz_width, qaz + half_quartz_height + 0.03] # Bottom-left corner home
                           ])
    
    return back_coords, mouth_coords, mirror_coords, mcp_coords

def visulise_tracks(sim_data):
    tracks_list, _, quartz_position, quartz_params = sim_data 
    quartz_depth, quartz_height, quartz_width = quartz_params
    qax, qay, qaz = quartz_position
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create Quartz Radiator
    verts = define_quartz(quartz_params, quartz_position)
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.20))
    
    # Plot tracks
    for track in tracks_list:
        track_id, init_vertex, quartz_in_vtx, quartz_out_vtx = track.get_track_data()
        # stack vertices vectors together into one array 
        vertices = np.vstack((init_vertex, quartz_in_vtx, quartz_out_vtx))
        # plot track
        ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], label=f"Track {track_id}")
    set_label(ax)
    ax.set_xlim(0, qax + (quartz_depth*3))
    ax.set_ylim(-(quartz_width*1.4), qay + (quartz_width*1.4))
    ax.set_zlim(qaz-quartz_height, qaz+quartz_height)
    #plt.legend()
    plt.show()

def visulise_radiation(sim_data):
    tracks_list, cherenkov_track_list, quartz_position, quartz_params = sim_data      
    quartz_depth, quartz_height, quartz_width = quartz_params
    qax, qay, qaz = quartz_position
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create Quartz Radiator
    verts = define_quartz(quartz_params, quartz_position)
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.1))
    
    # plot scatter points on cherenkov photon origin verticies
    for cherenkov_track in cherenkov_track_list:
        track_id, daughter_id, reflection_vertex_list = cherenkov_track.get_track_data()
        vertex = reflection_vertex_list[0]
        #ax.scatter(vertex[0], vertex[1], vertex[2], c='g', marker='o', s=1, alpha=0.5)
        direction_vec = cherenkov_track.radiator_exit_direction_vector
        # plot line from vertex following direction vector    
        ax.plot([vertex[0], vertex[0] + (direction_vec[0]*quartz_depth)/10], 
                [vertex[1], vertex[1] + (direction_vec[1]*quartz_width)/10],
                [vertex[2], vertex[2] + (direction_vec[2]*quartz_height)/10], c='g', alpha=0.3)


    # Plot tracks
    for track in tracks_list:
        if track.track_hit_radiator:
            track_id, init_vertex, quartz_in_vtx, quartz_out_vtx = track.get_track_data()
            # mark impacts
            ax.scatter(quartz_in_vtx[0], quartz_in_vtx[1], quartz_in_vtx[2], c='r', marker='o', s=1, alpha=0.5)
            ax.scatter(quartz_out_vtx[0], quartz_out_vtx[1], quartz_out_vtx[2], c='r', marker='o', s=1, alpha=0.5)
            # plot track
            vertices = np.vstack((init_vertex, quartz_in_vtx, quartz_out_vtx))
            ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', alpha=0.4, label=f"Track {track_id}")
    set_label(ax)   
    ax.set_xlim(qax, qax + (quartz_depth))
    ax.set_ylim(qay-(quartz_width/2), qay + (quartz_width/2))
    ax.set_zlim(qaz-(quartz_height/2), qaz+(quartz_height/2))        

    plt.show()

def visulise_quartz(sim_data):  
    tracks_list, cherenkov_track_list, quartz_position, quartz_params = sim_data      
    quartz_depth, quartz_height, quartz_width = quartz_params
    qax, qay, qaz = quartz_position
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create Quartz Radiator
    verts = define_quartz(quartz_params, quartz_position)
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.20))
    


    # Plot tracks
    for track in tracks_list:
        if track.track_hit_radiator:
            track_id, init_vertex, quartz_in_vtx, quartz_out_vtx = track.get_track_data()
            # mark impacts
            ax.scatter(quartz_in_vtx[0], quartz_in_vtx[1], quartz_in_vtx[2], c='r', marker='o', s=1, alpha=0.5)
            ax.scatter(quartz_out_vtx[0], quartz_out_vtx[1], quartz_out_vtx[2], c='r', marker='o', s=1, alpha=0.5)
            # plot track
            vertices = np.vstack((init_vertex, quartz_in_vtx, quartz_out_vtx))
            ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', alpha=0.4, label=f"Track {track_id}")
        
    # plot cherenkov photon tracks
    for cherenkov_track in cherenkov_track_list:
        track_id, daughter_id, reflection_vertex_list = cherenkov_track.get_track_data()

        reflection_vertex_list = np.array(reflection_vertex_list)
        radiator_exit_vertex = reflection_vertex_list[-1]
        radiator_exit_direction_vector = cherenkov_track.radiator_exit_direction_vector
        # plot track
        ax.plot(reflection_vertex_list[:, 0], reflection_vertex_list[:, 1], reflection_vertex_list[:, 2], c='g', alpha=0.4, label=f"Daghter {daughter_id} of Track {track_id}")

        ### note: fix this error!!!!!!!
        """
        # plot a short line that extends from the radiator exit vertex in the direction of the radiator exit direction vector
        ax.plot([radiator_exit_vertex[0], radiator_exit_vertex[0] + (radiator_exit_direction_vector[0]*quartz_depth)/2],
                [radiator_exit_vertex[1], radiator_exit_vertex[1] + (radiator_exit_direction_vector[1]*quartz_width)/2],
                [radiator_exit_vertex[2], radiator_exit_vertex[2] + (radiator_exit_direction_vector[2]*quartz_height)/2], c='g', alpha=0.9)
        """

    set_label(ax)      
    ax.set_xlim(qax - (quartz_depth*2), qax + (quartz_depth*3))
    ax.set_ylim(qay-(quartz_width*1.4), qay + (quartz_width*1.4))
    ax.set_zlim(qaz-quartz_height, qaz+quartz_height)

    plt.show()

def visulise_radiator_mouth(sim_data):   
    _, cherenkov_photons_list, quartz_position, quartz_params = sim_data   
    quartz_depth, quartz_height, quartz_width = quartz_params
    qax, qay, qaz = quartz_position
    half_quartz_width = quartz_width / 2
    half_quartz_height = quartz_height / 2
    back_coords, mouth_coords, mirror_coords, mcp_coords = set_detector_object_positions(quartz_params, quartz_position)    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create top of Quartz Radiator
    verts = define_quartz(quartz_params, quartz_position)
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=0.4, edgecolors='r', alpha=.06))
    
    # plot line between each corner
    #ax.plot(back_coords[:, 0], back_coords[:, 1], back_coords[:, 2], c='b', alpha=0.9)

    # Define vertices of front side of radiator that extends up into the focusing block (~3cm)
    #far_corner_coords = mouth_coords[:2]

    # plot line between each of the top corners and the corresponding oppsoite side corner on the far side of the cube
    #ax.plot([back_coords[2, 0], far_corner_coords[1, 0]], [back_coords[2, 1], far_corner_coords[1, 1]], [back_coords[2, 2], far_corner_coords[1, 2]], c='b', alpha=0.9)
    #ax.plot([back_coords[3, 0], far_corner_coords[0, 0]], [back_coords[3, 1], far_corner_coords[0, 1]], [back_coords[3, 2], far_corner_coords[0, 2]], c='b', alpha=0.9)

   
    ##### TRACKS #####
    # plot cherenkov photon tracks
    #for cherenkov_track in cherenkov_photons_list:
        #track_id, daughter_id, reflection_vertex_list = cherenkov_track.get_track_data()

        #reflection_vertex_list = np.array(reflection_vertex_list)
        #radiator_exit_vertex = reflection_vertex_list[-1]
        #radiator_exit_direction_vector = cherenkov_track.radiator_exit_direction_vector

        # plot line betwwen each track point
        #ax.plot(reflection_vertex_list[:, 0], reflection_vertex_list[:, 1], reflection_vertex_list[:, 2], c='g', alpha=0.2)

    # plot cherenkov photons final positions
    for cherenkov_track in cherenkov_photons_list:
        track_id, daughter_id, reflection_vertex_list = cherenkov_track.get_track_data()
        radiator_exit_vertex = cherenkov_track.radiator_exit_vertex
        #ax.scatter(radiator_exit_vertex[0], radiator_exit_vertex[1], radiator_exit_vertex[2], c='g', alpha=0.4)

        mouth_exit_vertex = cherenkov_track.radiator_mouth_exit_vertex
        ax.scatter(mouth_exit_vertex[0], mouth_exit_vertex[1], mouth_exit_vertex[2], c='r', alpha=0.7)
        
        # plot a short line that extends from the radiator exit vertex to the mouth exit vertex
        ax.plot([radiator_exit_vertex[0], mouth_exit_vertex[0]],
                [radiator_exit_vertex[1], mouth_exit_vertex[1]],
                [radiator_exit_vertex[2], mouth_exit_vertex[2]], c='b', alpha=0.4)

        
    set_label(ax)   
    ax.set_xlim(qax , qax + (quartz_depth))
    ax.set_ylim(qay-(quartz_width/2), qay + (quartz_width/2))
    ax.set_zlim(1.25, 1.28)
    #ax.view_init(azim=180, elev=0)
    plt.show()

def create_mirror(mirror_position):
        
    def make_curve(mirror_position):
        # Define parameters for the circle segment perimeter
        center = mirror_position
        radius = 0.26
        focal_length = radius / 2
        start_angle = 31.5  # in degrees
        end_angle = 51.8   # in degrees
        num_points = 1000  # Number of points to approximate the perimeter

        # Draw the circle segment perimeter in 3D
        angles = np.linspace(start_angle, end_angle, num_points)
        
        #points = np.vstack([center + np.array([radius * np.cos(np.radians(a)), radius * np.sin(np.radians(a)), 0]) for a in angles]) # aligned in z
        #points = np.vstack([center + np.array([0, radius * np.cos(np.radians(a)), radius * np.sin(np.radians(a))]) for a in angles]) # aligned in x
        points = np.vstack([center + np.array([radius * np.cos(np.radians(a)), 0, radius * np.sin(np.radians(a))]) for a in angles]) # aligned in y
        
        return points
    
    points1 = make_curve(np.array([2.0-0.08, -0.33, 1.25 - 0.055])) ### DEFINE FROM ARGUMENT INSTEAD 
    points2 = make_curve(np.array([2.0-0.08, .33, 1.25 - 0.055]))

    return points1, points2

def visulise_focusing_block(sim_data):
    """
    Plots the full focusing block and the tracks of the cherenkov photons as they exit the mouth of the radiator and hit the mirror, and then the mcp
    """
    # Defining variables     
    _, cherenkov_photons_list, quartz_position, quartz_params = sim_data 
    quartz_depth, quartz_height, quartz_width = quartz_params
    qax, qay, qaz = quartz_position
    half_quartz_width = quartz_width / 2
    half_quartz_height = quartz_height / 2
    back_coords, mouth_coords, mirror_coords, mcp_coords = set_detector_object_positions(quartz_params, quartz_position)
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create top of Quartz Radiator
    verts = define_quartz(quartz_params, quartz_position)
    ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=0.4, edgecolors='r', alpha=.06))
    
    # Draw lines between each corner to show back of radiator
    #ax.plot(back_coords[:, 0], back_coords[:, 1], back_coords[:, 2], c='b', alpha=0.9)

    # Define vertices of front side of radiator that extends up into the focusing block (~3cm)
    #far_corner_coords = mouth_coords[:2]

    # plot line between each of the top corners and the corresponding oppsoite side corner on the far side of the cube
    #ax.plot([back_coords[2, 0], far_corner_coords[1, 0]], [back_coords[2, 1], far_corner_coords[1, 1]], [back_coords[2, 2], far_corner_coords[1, 2]], c='b', alpha=0.9)
    #ax.plot([back_coords[3, 0], far_corner_coords[0, 0]], [back_coords[3, 1], far_corner_coords[0, 1]], [back_coords[3, 2], far_corner_coords[0, 2]], c='b', alpha=0.9)


    ##### MIRROR #####
    # plot lines bewteeen each corner in mirror_coords
    ax.plot(mirror_coords[:, 0], mirror_coords[:, 1], mirror_coords[:, 2], c='y', alpha=0.9)

    """
    # PLOT MIRROR curves
    points1, points2 = create_mirror(mirror_coords)
    for i in range(len(points1) - 1):
        ax.plot([points1[i][0], points1[i + 1][0]], 
                [points1[i][1], points1[i + 1][1]], 
                [points1[i][2], points1[i + 1][2]], color="black")

    for i in range(len(points2) - 1):
        ax.plot([points2[i][0], points2[i + 1][0]], 
                [points2[i][1], points2[i + 1][1]], 
                [points2[i][2], points2[i + 1][2]], color="black")

    #PLOT LINES BETWEEN first two corner coordinates
    ax.plot([mirror_coords[0][0], mirror_coords[1][0]],
            [mirror_coords[0][1], mirror_coords[1][1]],
            [mirror_coords[0][2], mirror_coords[1][2]], c='b', alpha=0.4)

    # plot line between last two corner coordinates
    ax.plot([mirror_coords[2][0], mirror_coords[3][0]],
            [mirror_coords[2][1], mirror_coords[3][1]],
            [mirror_coords[2][2], mirror_coords[3][2]], c='b', alpha=0.4)

    """


    
    ##### MCP #####
    # plot line between each corner
    ax.plot(mcp_coords[:, 0], mcp_coords[:, 1], mcp_coords[:, 2], c='y', alpha=0.9)

    
    ##### TRACKS #####
    """
    # plot cherenkov photon tracks
    for cherenkov_track in cherenkov_photons_list:
        track_id, daughter_id, reflection_vertex_list = cherenkov_track.get_track_data()

        reflection_vertex_list = np.array(reflection_vertex_list)
        radiator_exit_vertex = reflection_vertex_list[-1]
        radiator_exit_direction_vector = cherenkov_track.radiator_exit_direction_vector

        # plot line betwwen each track point
        #ax.plot(reflection_vertex_list[:, 0], reflection_vertex_list[:, 1], reflection_vertex_list[:, 2], c='g', alpha=0.2)
    """
    # plot cherenkov photons final positions
    for cherenkov_track in cherenkov_photons_list:
        track_id, daughter_id, reflection_vertex_list = cherenkov_track.get_track_data()
        radiator_exit_vertex = cherenkov_track.radiator_exit_vertex
        #radiator_exit_direction_vector = cherenkov_track.radiator_exit_direction_vector
        #ax.scatter(radiator_exit_vertex[0], radiator_exit_vertex[1], radiator_exit_vertex[2], c='g', alpha=0.6)
        
        mouth_exit_vertex = cherenkov_track.radiator_mouth_exit_vertex
        #mouth_exit_direction_vector = cherenkov_track.radiator_mouth_exit_direction_vector
        #ax.scatter(mouth_exit_vertex[0], mouth_exit_vertex[1], mouth_exit_vertex[2], c='r', alpha=0.9)
        """
        # plot a short line that extends from the radiator exit vertex to the mouth exit vertex
        ax.plot([radiator_exit_vertex[0], mouth_exit_vertex[0]],
                [radiator_exit_vertex[1], mouth_exit_vertex[1]],
                [radiator_exit_vertex[2], mouth_exit_vertex[2]], c='b', alpha=0.9)
        """
        # plot line from mouth exit vertex to the mirror intersection
        ax.plot([mouth_exit_vertex[0], cherenkov_track.mirror_reflection_coordinate[0]],
                [mouth_exit_vertex[1], cherenkov_track.mirror_reflection_coordinate[1]],
                [mouth_exit_vertex[2], cherenkov_track.mirror_reflection_coordinate[2]], c='b', alpha=0.9)
        
        """
        # plot short line that extends from the mirror intersection following the direction vector
        ax.plot([cherenkov_track.mirror_reflection_coordinate[0], cherenkov_track.mirror_reflection_coordinate[0] + cherenkov_track.mirror_reflection_direction_vector[0]],
                [cherenkov_track.mirror_reflection_coordinate[1], cherenkov_track.mirror_reflection_coordinate[1] + cherenkov_track.mirror_reflection_direction_vector[1]],
                [cherenkov_track.mirror_reflection_coordinate[2], cherenkov_track.mirror_reflection_coordinate[2] + cherenkov_track.mirror_reflection_direction_vector[2]], c='b', alpha=0.9)

        """

        # plot line from mirror intersection to the mcp intersection
        ax.plot([cherenkov_track.mirror_reflection_coordinate[0], cherenkov_track.mcp_global_coordinate[0]],
                [cherenkov_track.mirror_reflection_coordinate[1], cherenkov_track.mcp_global_coordinate[1]],
                [cherenkov_track.mirror_reflection_coordinate[2], cherenkov_track.mcp_global_coordinate[2]], c='g', alpha=0.9)
        
        # plot scatter points at the mirror and mcp intersection
        ax.scatter(cherenkov_track.mirror_reflection_coordinate[0], cherenkov_track.mirror_reflection_coordinate[1], cherenkov_track.mirror_reflection_coordinate[2], c='r', alpha=0.9)
        ax.scatter(cherenkov_track.mcp_global_coordinate[0], cherenkov_track.mcp_global_coordinate[1], cherenkov_track.mcp_global_coordinate[2], c='r', alpha=0.9)

    set_label(ax)    
    ax.set_xlim(1.95, 2.15)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(qaz+(quartz_height/2)-0.05, 1.4)
    #ax.set_zlim(qaz+(quartz_height/2), qaz+(quartz_height/2))

    # set plot view
    ax.view_init(azim=270, elev=0)
    plt.show()

def visulise_MCP_global(sim_data):   
    """
    3D plots of the MCP array in the global coordinate system.
    First plot is an angled view of the MCP array.
    Second plot is a face on view of the MCP array.
    """
    _, cherenkov_photons_list, quartz_position, quartz_params = sim_data   
    quartz_depth, quartz_height, quartz_width = quartz_params
    qax, qay, qaz = quartz_position
    half_quartz_width = quartz_width / 2
    half_quartz_height = quartz_height / 2
    _, _, _, mcp_coords = set_detector_object_positions(quartz_params, quartz_position)
    
    fig = plt.figure(figsize=(10, 20))
    ax = fig.add_subplot(122, projection='3d')
    ax2 = fig.add_subplot(121, projection='3d')

    # plot line between each corner
    ax.plot(mcp_coords[:, 0], mcp_coords[:, 1], mcp_coords[:, 2], c='b', alpha=0.9)
    ax2.plot(mcp_coords[:, 0], mcp_coords[:, 1], mcp_coords[:, 2], c='b', alpha=0.9)

    # plot scatte rpoint for each mcp intersection
    for cherenkov_track in cherenkov_photons_list:
        ax.scatter(cherenkov_track.mcp_global_coordinate[0], cherenkov_track.mcp_global_coordinate[1], cherenkov_track.mcp_global_coordinate[2], c='r', alpha=0.9)
        ax2.scatter(cherenkov_track.mcp_global_coordinate[0], cherenkov_track.mcp_global_coordinate[1], cherenkov_track.mcp_global_coordinate[2], c='r', alpha=0.9)
    
    for axes in ([ax, ax2]):
        set_label(ax)
        axes.set_xlim(qax, qax-0.04)
        axes.set_ylim(qay - half_quartz_width, qay + half_quartz_width)
        axes.set_zlim(qaz + half_quartz_height + 0.08, qaz + half_quartz_height + 0.03)

    # calculate the angle betweeen z= (qaz + half_quartz_height + 0.08) and z= (qaz + half_quartz_height + 0.03)
    x1, y1 = qax, qax-0.04
    x2, y2 = qaz + half_quartz_height + 0.08, qaz + half_quartz_height + 0.03
    angle_rad = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle_rad) + 180

    ax.set_title("MCP View Face On, Global CS")
    ax2.set_title("MCP View Angled, Global CS")
    # set plot view
    ax.view_init(azim=-180-180, elev=angle_deg+90-180-90)
    ax2.view_init(azim=30, elev=210)
    plt.show()

def visulise_MCP_local(sim_data):
    """
    2D plot of the MCP array in the local coordinate system of the MCP and in mm rather than m.
    """
    _, cherenkov_track_list, quartz_position, quartz_params = sim_data
    quartz_depth, quartz_height, quartz_width = quartz_params
    qax, qay, qaz = quartz_position
    half_quartz_width = quartz_width / 2
    half_quartz_height = quartz_height / 2

    # Define the coordinates of the corners of the initial rectangle ( CHANGED UNITS TO mm FROM m IN THE GLOBAL SYSTEM)
    x0, y0 = -330.0, -30.0  # Bottom-left corner #mm
    x1, y1 = 330.0, 30.0  # Top-right corner #mm

    # The size of the active area (53 x 53 mm) and the spacing between them derived from the pitch (mcp pitch = 60 x 60 mm)
    square_size = 53.0 # mm
    spacing = 3.5 # mm

    # create figure
    fig = plt.figure(figsize=(16, 5))
    ax = fig.add_subplot(111)

    # Plot the initial rectangle
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='blue', linewidth=2))

    # Plot the individual MCP's
    for i in range(11):
        x = x0 + spacing + i * (square_size + 2 * spacing)
        for j in range(1):
            y = y0 + spacing + j * (square_size + 2 * spacing)
            ax.add_patch(plt.Rectangle((x, y), square_size, square_size, fill=False, color='blue'))

    # plot scatter point for each mcp intersection
    for cherenkov_track in cherenkov_track_list:
        x, y, z = cherenkov_track.MCP_local_coordinate
        plt.scatter(z*1000, y*1000, c='r', alpha=0.9) # vals * 1000 to convert m to mm

    # Set axis limits
    #ax.set_xlim(x0, x1)
    #ax.set_ylim(y0, y1)

    # Show the plot
    #plt.gca().set_aspect('equal', adjustable='box')  # Equal aspect ratio
    #plt.axis('off')  # Turn off axis

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("MCP View Face On, Local CS")

    plt.show()

# %%
class ParticleTrack():
    def __init__(self, track_id, energy, init_vertex, init_dir_vector, quartz_position, quartz_params, particle_type):
        self.track_id = track_id  # Unique identifier for the track
        self.particle_type = particle_type  # Type of particle (e.g., electron, proton, etc.)
        self.energy = energy  # Energy of the particle track in GeV/c
        self.velocity = 0.9  # Velocity in terms of the speed of light #np.sqrt(1 - (1 / (self.energy + 1))**2)  # Velocity of the particle track
        self.init_vertex = np.array(init_vertex)  # Starting position of the particle track (x, y, z)
        self.init_dir_vector = np.array(init_dir_vector)  # Ending position of the particle track (x, y, z)
        self.direction_vector = self.init_dir_vector  # Direction vector of the particle track (gets updated)
        self.distance_to_quartz = quartz_position[0] # Quartz front face straight line distance from 0, 0, 0
        self.quartz_depth = quartz_params[0] # Quartz depth
        self.quartz_height = quartz_params[1] # Quartz height
        self.quartz_width = quartz_params[2] # Quartz width
        self.propegate_track()

    def propegate_track(self):
        """
        Propagate the particle track through the detector by taking starting position and 
        direction vector and then finding intersection with detector quartz
        """

        # Normalise dir vector
        norm_dir_vector = self.init_dir_vector / np.linalg.norm(self.init_dir_vector)

        # using trig find the coordinate on the detector face which is a plane that lies at (distance_to_quartz, 0, 0) from the start point of init_vertex following the direction of norm_dir_vector
        #in vtx
        x = self.distance_to_quartz
        y = x * norm_dir_vector[1] / norm_dir_vector[0]
        z = x * norm_dir_vector[2] / norm_dir_vector[0]

        # calculate length of track befor hitting detector face
        length = np.sqrt(x**2 + y**2 + z**2)
        
        # check if track hits detector face
        if y < self.quartz_width/2 and y > -self.quartz_width/2 and z < self.quartz_height/2 and z > -self.quartz_height/2:
            self.quartz_in_vtx = np.array((x, y, z))
            self.tracklength = length

            # using trig find the coordinate on the detector face which is a plane that lies at (distance_to_quartz, 0, 0) from the start point of init_vertex following the direction of norm_dir_vector
            #out vtx
            x = self.distance_to_quartz + self.quartz_depth
            y = x * norm_dir_vector[1] / norm_dir_vector[0]
            z = x * norm_dir_vector[2] / norm_dir_vector[0]
            self.quartz_out_vtx = np.array((x, y, z))

            # calculate length of track inside quartz radiator
            length2 = np.sqrt(x**2 + y**2 + z**2)
            self.radiator_path_length = length2 - length
            self.track_hit_radiator = True
        
        else:
            self.quartz_in_vtx = np.array((x, y, z))
            self.quartz_out_vtx = np.array((np.nan, np.nan, np.nan))
            self.tracklength = np.nan
            self.radiator_path_length = np.nan
            self.track_hit_radiator = False


    def cherenkov_angles(self, direction_vector, track_velocity, num_vectors, delta_cherenkov_angle, max_magnitude, refractive_index=1.33):
        cherenkov_angle = np.arccos(1/(track_velocity*refractive_index)) #radiens  # simplified from (c/(v*refractive_index)) as v is is units of c already

        # Generate random values for spherical coordinates
        theta = np.random.uniform(0, 2 * np.pi, num_vectors)  # Azimuthal angle
        phi = np.random.uniform(180 - cherenkov_angle, 180 - (cherenkov_angle + delta_cherenkov_angle), num_vectors)  # Polar angle within the cone

        # Calculate the Cartesian coordinates
        x = max_magnitude * np.sin(np.radians(phi)) * np.cos(theta)
        y = max_magnitude * np.sin(np.radians(phi)) * np.sin(theta)
        z = max_magnitude * np.cos(np.radians(phi))

        # Create the random vectors
        random_vectors = np.column_stack((x, y, z))

        reference_vector =  direction_vector
        target_vector = np.array([0, 0, 1])

        # Calculate the rotation matrix using the cross product
        cross_product = np.cross(target_vector, reference_vector)
        dot_product = np.dot(target_vector, reference_vector)

        # Create the rotation matrix
        skew_matrix = np.array([[0, -cross_product[2], cross_product[1]],
                                [cross_product[2], 0, -cross_product[0]],
                                [-cross_product[1], cross_product[0], 0]])

        rotation_matrix = np.eye(3) + skew_matrix + \
                        np.dot(skew_matrix, skew_matrix) * (1 - dot_product) / np.linalg.norm(cross_product) ** 2

        # Apply the rotation to all vectors
        rotated_vectors = np.dot(rotation_matrix, random_vectors.T).T

        return rotated_vectors



    def radiate_cherenkov(self, medium_refractive_index, number_of_photons, delta_cherenkov_angle):
        # Create cherenkov photons origin vertex list
        t = np.random.random(number_of_photons)
        self.cherenkov_photons_origin_vertex_list = self.quartz_in_vtx + t[:, np.newaxis] * (self.quartz_out_vtx - self.quartz_in_vtx)

        # Create cherenkov photons dir vectors list
        max_magnitude = 1.0  # Maximum magnitude of the vectors
        self.cherenkov_photons_dir_vectors_list = self.cherenkov_angles(self.direction_vector, self.velocity, number_of_photons, delta_cherenkov_angle, max_magnitude, medium_refractive_index)

        #return self.cherenkov_photons_origin_vertex_list, self.cherenkov_photons_dir_vectors_list

    def get_track_data(self):
        return self.track_id, self.init_vertex, self.quartz_in_vtx, self.quartz_out_vtx
    
    def print_info(self):
        # Print information about the particle track
        print(f"Track ID: {self.track_id}")
        print(f"Particle Type: {self.particle_type}")
        print(f"Energy: {self.energy} GeV/c")
        print(f"Initial Vertex: {self.init_vertex}")
        print(f"Initial Direction Vector: {self.init_dir_vector}")
        print(f"Distance to Quartz: {self.distance_to_quartz} m")
        print(f"Quartz Depth: {self.quartz_depth} m")
        print(f"Track Length: {self.tracklength} m")
        print(f"Track Length in Quartz: {self.radiator_path_length} m")
        print(f"Quartz Near Side Intersection Vertex: {self.quartz_in_vtx}")
        print(f"Quartz Far Side Intersection Vertex: {self.quartz_out_vtx}")
        print(f"Track Hit Quartz: {self.track_hit_radiator}")
        print("\n")

# %%
class CherenkovPhoton():
    def __init__(self, track_id, daughter_track_id, energy, init_vertex, init_dir_vector, quartz_params, quartz_position):
        # start timer
        start_time = time.time()
        self.track_id = track_id  # Unique identifier for the mother particle track
        self.daughter_track_id = daughter_track_id  # Unique identifier for the cherenkov photon track
        self.particle_type = "photon"  # Type of particle (e.g., electron, proton, etc.)
        self.energy = energy  # Energy of the photon
        #self.init_vertex = init_vertex  # Starting position of the particle track (x, y, z)
        init_dir_vector1 = np.random.uniform(size=3) # init_dir_vector #/ np.linalg.norm(init_dir_vector) 
        self.init_dir_vector =  init_dir_vector1 / np.linalg.norm(init_dir_vector1) 
        self.init_dir_vector = init_dir_vector
        self.quartz_params = quartz_params
        self.quartz_position = quartz_position       
        self.back_coords, self.mouth_coords, self.mirror_coords, self.mcp_coords = set_detector_object_positions(quartz_params, quartz_position)
        self.radiator_mouth_exit_vertex = np.array([np.nan, np.nan, np.nan])
        self.radiator_mouth_exit_direction_vector = np.array([np.nan, np.nan, np.nan])
        self.mirror_reflection_coordinate = np.array([np.nan, np.nan, np.nan]) # Mirror reflection coordinate
        self.mirror_reflection_direction_vector = np.array([np.nan, np.nan, np.nan]) # Mirror reflection direction vector
        self.mcp_global_coordinate = np.array([np.nan, np.nan, np.nan]) # MCP reflection coordinate
        self.MCP_local_coordinate = np.array([np.nan, np.nan, np.nan])
        self.left_radiator = False
        self.left_mouth = False
        self.hit_mirror = False
        self.hit_mcp = False
        init_time = time.time() - start_time
        print(f"Time to initialise CherenkovPhoton: {init_time} seconds")
        #self.critical_angle = self.critical_angle_snells_law(n1=1.5, n2=1) # CONNECT
        self.propegate_photon(init_vertex, init_dir_vector)
        pp_time = time.time() - init_time
        print(f"Time to propegate photon: {pp_time} seconds")
        self.propegate_through_focusing_block()
        fb_time = time.time() - pp_time
        print(f"Time to propegate photon through focusing block: {fb_time} seconds")
        self.readout_MCP()
        mcp_time = time.time() - fb_time
        print(f"Time to readout MCP: {mcp_time} seconds")

    def critical_angle_snells_law(self, n1, n2):
        """
        Calculate the critical angle for total internal reflection.
        
        Args:
        n1 (float): Refractive index of the first medium.
        n2 (float): Refractive index of the second medium.
        
        Returns:
        float: The critical angle in degrees.
        """     
        # Calculate the critical angle in radians using Snell's Law.
        critical_angle_rad = np.arcsin(n2 / n1)  # changgy to nunpy aSIN
        
        # Convert the angle from radians to degrees.
        critical_angle_deg = np.degrees(critical_angle_rad)   #change to numpy function
        
        return critical_angle_deg



    def print_info(self):
        print(f"Track ID: {self.track_id}")
        print(f"Daughter Track ID: {self.daughter_track_id}")
        print(f"Particle Type: {self.particle_type}")
        print(f"Energy: {self.energy} MeV")
        print(f"Initial Vertex: {self.init_vertex}")
        print(f"Initial Direction Vector: {self.init_dir_vector}")
        print(f"Surface Intersections: {self.surface_intersections_list}")
        print(f"Radiator Exit Vertex: {self.radiator_exit_vertex}")
        print(f"Radiator Exit Direction Vector: {self.radiator_exit_direction_vector}")
        print(f"Radiator Path Length: {self.radiator_path_length}")
        print("\n")
        
    def get_track_data(self):
        return self.track_id, self.daughter_track_id, self.surface_intersections_list
    
    def propegate_photon(self, origin, direction_vector):
        quartz_depth, quartz_height, quartz_width = self.quartz_params
        qax, qay, qaz = self.quartz_position
        x_dim, y_dim, z_dim = quartz_depth, quartz_width, quartz_height
        position = np.array(origin, dtype=float)
        direction = np.array(direction_vector, dtype=float)
        data = [origin]
        while position[2] < (z_dim / 2) - qaz:  # Exit via top of reflector
            
            distances = np.array([(x_dim - (position[0] - qax)) / direction[0] if direction[0] > 0 else (position[0] - qax) / -direction[0],
                                 ((qay + y_dim / 2) - position[1]) / direction[1] if direction[1] > 0 else (position[1] - (qay - y_dim / 2)) / -direction[1],
                                 ((qaz + z_dim / 2) - position[2]) / direction[2] if direction[2] > 0 else (position[2] - (qaz - z_dim / 2)) / -direction[2]])

            # Find the minimum positive distance and corresponding side
            min_distance = np.min(distances)
            min_distance_indices = np.where(np.isclose(distances, min_distance))[0]
            
            # Update position based on the minimum distance
            position = position + min_distance * direction
            data.append(position.copy())

            if position[2] < (z_dim / 2) - qaz:  ### CLEANUP, protection from direction vctor changing for the output face due to the last interaction
                # Reflect the direction vector based on the side hit
                for min_distance_index in min_distance_indices:
                    if min_distance_index == 0:
                        direction[0] *= -1
                    elif min_distance_index == 1:
                        direction[1] *= -1
                    elif min_distance_index == 2:
                        direction[2] *= -1

        #return position dir_vec, radiator_path_data, radiator_path_length
        self.surface_intersections_list = data
        self.radiator_exit_vertex = position
        self.radiator_exit_direction_vector = direction
        self.left_radiator = True

        # length is distance from across all the paths between intersections
        self.radiator_path_length = np.sum(np.linalg.norm(np.diff(data, axis=0), axis=1))  ##### SHOULD THIS INCLUDE PATH IN THE FOCUSING BLOCK>?????

    def line_plane_intersection(self, origin, direction_vector, plane_corners):
        plane_normal = np.cross(plane_corners[1] - plane_corners[0], plane_corners[3] - plane_corners[0])
        
        # Check if photon is parallel to the plane
        if np.allclose(np.dot(direction_vector, plane_normal), 0):
            return None
        
        # Determine the orientation of the plane based on the photon direction
        if np.dot(direction_vector, plane_normal) < 0:
            plane_normal = -plane_normal
        
        t = np.dot(plane_corners[0] - origin, plane_normal) / np.dot(direction_vector, plane_normal)
        
        intersection_point = origin + t * direction_vector
        
        min_x, max_x = np.min(plane_corners[:, 0]), np.max(plane_corners[:, 0])
        min_y, max_y = np.min(plane_corners[:, 1]), np.max(plane_corners[:, 1])
        
        if min_x <= intersection_point[0] <= max_x and min_y <= intersection_point[1] <= max_y:
            return intersection_point
        
        else: # No intersection photon missed plane
            return None

    def propegate_through_focusing_block(self):
        qax, qay, qaz = self.quartz_position
        quartz_depth, quartz_height, quartz_width = self.quartz_params
        half_quartz_width = quartz_width / 2
        half_quartz_height = quartz_height / 2


        ### BACK WALL OF FOCUSING BLOCK ###
        # Check for reflection in the focusing block mouth off the back wall
        fbb_intersection = self.line_plane_intersection(self.radiator_exit_vertex, self.radiator_exit_direction_vector, self.back_coords)
        if fbb_intersection is not None:
            # Update position and direction vector
            position = fbb_intersection
            self.radiator_exit_direction_vector[0] *= -1
            self.surface_intersections_list.append(position)
            self.radiator_exit_vertex = position

        ### SIDE WALLS OF FOCUSING BLOCK ###
        # Check for reflection in the focusing block mouth off the side walls. SHOULD THESE BE DISCAREDED OR REFLECTED?????? I GUES REFLECTED AND A SOURCE OF POTENTIAL NOISE??
        #CODE#

        ### MOUTH OF FOCUSING BLOCK ###
        fbm_intersection = self.line_plane_intersection(self.radiator_exit_vertex, self.radiator_exit_direction_vector, self.mouth_coords)
        if fbm_intersection is not None:
            # Update position and direction vector
            position = fbm_intersection
            self.surface_intersections_list.append(position)
            self.radiator_mouth_exit_vertex = position
            self.radiator_mouth_exit_direction_vector = self.radiator_exit_direction_vector
            self.left_mouth = True
        else:
            #print("No mouth of focusing block intersection")
            pass

        ### MIRROR OF FOCUSING BLOCK ###
        mirror_intersection = self.line_plane_intersection(self.radiator_mouth_exit_vertex, self.radiator_mouth_exit_direction_vector, self.mirror_coords)
        if mirror_intersection is not None:
            # Update position and direction vector
            self.surface_intersections_list.append(mirror_intersection)
            self.mirror_reflection_coordinate = mirror_intersection
            self.mirror_reflection_direction_vector = self.mirror_reflection(self.radiator_mouth_exit_direction_vector, self.mirror_coords)   
            self.hit_mirror = True

            ### MCP ARRAY OF FOCUSING BLOCK ###
            mcp_intersection = self.line_plane_intersection(self.mirror_reflection_coordinate, self.mirror_reflection_direction_vector, self.mcp_coords)
            if mcp_intersection is not None:
                # Update position and direction vector
                position = mcp_intersection
                self.surface_intersections_list.append(position)
                self.mcp_global_coordinate = position
                self.hit_mcp = True

    def mirror_reflection(self, photon_direction, mirror_coords):

        #calulate the mirror normal vector from its corner coordinates
        mirror_normal = np.cross(mirror_coords[1] - mirror_coords[0], mirror_coords[3] - mirror_coords[0])
        mirror_normal = mirror_normal / np.linalg.norm(mirror_normal)

        # Calculate the dot product of photon direction and mirror normal
        dot_product = np.dot(photon_direction, mirror_normal)

        # Calculate the new direction vector after reflection
        reflected_direction = photon_direction - 2 * dot_product * mirror_normal

        return reflected_direction


    def readout_MCP(self):
        qax, qay, qaz = self.quartz_position
        quartz_depth, quartz_height, quartz_width = self.quartz_params
        half_quartz_width = quartz_width / 2
        half_quartz_height = quartz_height / 2


        def shift_coordinates_3d(coordinates, new_origin):
            shifted_coordinates = coordinates - np.array(new_origin)
            return shifted_coordinates

        ### Transform origin to centre of mcp
        centre = np.mean(self.mcp_coords[:4], axis=0)
        new_origin_3d = centre #np.array(centre, dtype=float)
        shifted_mcp = shift_coordinates_3d(self.mcp_coords, new_origin_3d)
        shifted_data = shift_coordinates_3d(self.mcp_global_coordinate, new_origin_3d)

        # Define the coordinates of the four corners of the square (A, B, C, and D)
        A = shifted_mcp[0]
        B = shifted_mcp[1]
        C = shifted_mcp[2]
        D = shifted_mcp[3]

        # Define the basis vectors of the new coordinate system
        U = (B - A) / np.linalg.norm(B - A)  # Normalize vector AB to get the x-axis vector U
        V = (D - A) / np.linalg.norm(D - A)  # Normalise vector AD to get the y-axis vector V
        N = np.cross(B - A, D - A)           # Calculate the normal vector N by taking the cross product of AB and AD
        new_basis_vectors = [N, V, U]         

        ### Transform the coordinates by 3d rotation to centre on mcp
        # Convert the basis vectors to a transformation matrix
        T = np.array(new_basis_vectors).T

        # Inverse of the transformation matrix to go from standard basis to new basis
        T_inv = np.linalg.inv(T)

        # Create an array of original coordinates as columns
        coords = np.array(shifted_data).T

        # Apply the transformation to the original coordinates
        transformed_coords = np.dot(T_inv, coords)

        # Transpose back to get the transformed coordinates as rows
        transformed_coords = transformed_coords.T

        self.MCP_local_coordinate = transformed_coords



# %%


def main():
    # Your entire script here


    # %%
    ## Example of generating multiple tracks and photons
    num_of_particles = 100
    n_cher_per_track = 30   # SHOULD BE 30 FOR REALISTIC SIMULATION DUE TO QUANTUM EFFICIENCY OF THE DETECTOR

    #Single torch module
    quartz_params = (0.01, 2.5, 0.66) # depth, height, width all in meters
    quartz_position = (2, 0, 0) # centre of front face
    quartz_refractive_index = 1.4496   # charged particle speed must be 1/n c   n='quartz_refractive_index' is condition for cherenkov radiation


    # %%
    # Initialize event counter
    event_counter = EventCounter(num_of_particles, n_cher_per_track)

    # Initialize lists to store tracks and photons
    tracks_list = []
    cherenkov_photons_list = []
    for i in range (num_of_particles):
        track_id = i + 1
        init_vertex = (0,0,0)#(0, np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
        init_dir_vector = (1, np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5))
        energy = 8.0 # GeV/c
        
        # Generate a track
        genrated_track = ParticleTrack(track_id, energy, init_vertex, init_dir_vector, quartz_position, quartz_params, particle_type="pion")
        tracks_list.append(genrated_track)
        
        # Check if the track hit the radiator
        if genrated_track.track_hit_radiator:
            event_counter.add_track()
            #print(f"\n \n    Track ID: {track_id}")
            # Generate cherenkov photon origins and direction vectors
            genrated_track.radiate_cherenkov(quartz_refractive_index, n_cher_per_track, delta_cherenkov_angle=0)
            origins = genrated_track.cherenkov_photons_origin_vertex_list
            vecs = genrated_track.cherenkov_photons_dir_vectors_list

            # Create cherenkov photons
            for dt_id, (cherenkov_photon_vert, cherenkov_photon_dir_vec) in enumerate(zip(origins, vecs)):
                photon = CherenkovPhoton(track_id, dt_id+1, 1, cherenkov_photon_vert, cherenkov_photon_dir_vec, quartz_params, quartz_position)
                cherenkov_photons_list.append(photon)
                event_counter.count_photons(photon)

    sim_data = [tracks_list, cherenkov_photons_list, quartz_position, quartz_params]


    # %%
    visulise_tracks(sim_data)

    visulise_radiation(sim_data)#, show_radiator=True, show_tracks=True, show_exit_vertex=True)

    visulise_quartz(sim_data)#, show_radiator=True, show_tracks=True, show_exit_vertex=True)

    visulise_radiator_mouth(sim_data)#, show_tracks=True, show_exit_vertex=True)

    visulise_focusing_block(sim_data)#, show_tracks=True, show_exit_vertex=True, show_mouth=True, show_mirror=True, show_MCP=True)

    visulise_MCP_global(sim_data)

    visulise_MCP_local(sim_data)

    # %%
    #### INVESTIGATING NUMBER OF HITS ON SURFACES
    event_counter.print_info()

# %%
import sys 
if __name__ == "__main__":
    with open("profiling_output.txt", "w") as output_file:
        sys.stdout = output_file  # Redirect stdout to the file
        cProfile.run("main()", sort='cumulative')
    sys.stdout = sys.__stdout__

# %%

#### INVESTIGATE THE REFLECTIONS IN THE FOCUSING BLOCK MOUTH AND SIDES



# %%


