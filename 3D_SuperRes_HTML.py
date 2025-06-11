import os
import numpy as np
import struct
import webbrowser
import plotly.graph_objects as go
from tkinter import Tk, filedialog, messagebox, simpledialog
from tqdm import tqdm

def select_stl_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select STL file",
        filetypes=[("STL files", "*.stl")]
    )
    return file_path

def load_stl(file_path):
    """Load a binary STL file."""
    with open(file_path, 'rb') as f:
        header = f.read(80)
        num_triangles_bytes = f.read(4)
        num_triangles = struct.unpack('<I', num_triangles_bytes)[0]
        expected_size = 84 + num_triangles * 50
        file_size = os.path.getsize(file_path)

    if expected_size <= file_size + 100:
        return load_binary_stl(file_path, num_triangles)
    else:
        raise ValueError("File doesn't appear to be standard binary STL.")

def load_binary_stl(file_path, num_triangles):
    """Load binary STL safely with progress bar."""
    vertices = []
    faces = []
    with open(file_path, 'rb') as f:
        f.seek(84)  # Skip header
        i = 0
        with tqdm(total=num_triangles, desc="Loading Binary STL", unit="triangles") as pbar:
            for _ in range(num_triangles):
                data = f.read(50)
                if len(data) < 50:
                    print("Warning: Unexpected end of file!")
                    break
                normal = struct.unpack('<3f', data[0:12])
                v1 = struct.unpack('<3f', data[12:24])
                v2 = struct.unpack('<3f', data[24:36])
                v3 = struct.unpack('<3f', data[36:48])
                vertices.extend([v1, v2, v3])
                faces.append([i, i+1, i+2])
                i += 3
                pbar.update(1)
    return np.array(vertices), np.array(faces)

def decimate_mesh(vertices, faces, target_percentage=0.3):
    """Smart decimation: preserve surface better."""
    print(f"🔵 Decimating: keeping {int(target_percentage * 100)}% of faces...")
    total_faces = len(faces)
    keep_faces = int(total_faces * target_percentage)
    
    if keep_faces < 3:
        raise ValueError("Too few faces after decimation!")

    np.random.seed(42)
    shuffled_indices = np.random.permutation(total_faces)
    selected_indices = shuffled_indices[:keep_faces]

    faces = faces[selected_indices]

    unique_vertices, inverse_indices = np.unique(vertices[faces.flatten()], axis=0, return_inverse=True)
    faces = inverse_indices.reshape((-1, 3))

    print(f"✅ Decimated to {len(faces)} faces and {len(unique_vertices)} vertices.")
    return unique_vertices, faces

def save_as_html(vertices, faces, output_html_path):
    """Save 3D STL mesh as interactive Plotly HTML."""
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    i = np.array(faces)[:, 0]
    j = np.array(faces)[:, 1]
    k = np.array(faces)[:, 2]

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                opacity=0.7,
                color='lightblue'
            )
        ]
    )

    fig.update_layout(
        title='Interactive 3D STL Model',
        scene=dict(
            xaxis_title='X (microns)',
            yaxis_title='Y (microns)',
            zaxis_title='Z (microns)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.write_html(output_html_path)
    print(f"✅ HTML saved: {output_html_path}")

    webbrowser.open(f'file://{os.path.abspath(output_html_path)}')

    root = Tk()
    root.withdraw()
    messagebox.showinfo("Success!", f"Interactive 3D HTML saved:\n{output_html_path}")

def stl_to_html():
    stl_file = select_stl_file()
    if not stl_file:
        print("❗ No file selected. Exiting.")
        return

    vertices, faces = load_stl(stl_file)

    root = Tk()
    root.withdraw()
    answer = messagebox.askyesno("Decimation?", "Reduce mesh for faster display?")

    if answer:
        percent = simpledialog.askfloat(
            "Decimation Percentage",
            "Enter % of faces to keep (e.g., 0.3 = 30%):",
            minvalue=0.05,
            maxvalue=1.0
        )
        if percent:
            vertices, faces = decimate_mesh(vertices, faces, target_percentage=percent)

    output_dir = os.path.dirname(stl_file)
    base_name = os.path.splitext(os.path.basename(stl_file))[0]
    output_html = os.path.join(output_dir, f"{base_name}_interactive.html")

    save_as_html(vertices, faces, output_html)

if __name__ == "__main__":
    stl_to_html()
