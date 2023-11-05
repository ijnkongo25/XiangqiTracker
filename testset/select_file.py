import os
import random

def randomly_keep_files(directory, num_to_keep=50):
    # List all files in the directory
    all_files = os.listdir(directory)

    # Check if the number of files is less than or equal to the number to keep
    if len(all_files) <= num_to_keep:
        print("Nothing to delete. The number of files is less than or equal to the specified limit.")
        return

    # Randomly shuffle the list of files
    random.shuffle(all_files)

    # Files to keep
    files_to_keep = all_files[:num_to_keep]

    # Files to delete
    files_to_delete = all_files[num_to_keep:]

    # Delete the files
    for file in files_to_delete:
        file_path = os.path.join(directory, file)
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {str(e)}")

    print(f"Kept {num_to_keep} files.")

# Specify the directory path and the number of files to keep
num_files_to_keep = 20
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\cannon_black"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\cannon_red"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\elephant_black"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\elephant_red"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\guard_black"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\guard_red"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\king_black"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\king_red"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\knight_black"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\knight_red"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\pawn_black"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\pawn_red"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\rook_black"
randomly_keep_files(directory_path, num_files_to_keep)
directory_path = "E:\AMME4710_major\CNN\\testset\\test\\rook_red"
randomly_keep_files(directory_path, num_files_to_keep)
