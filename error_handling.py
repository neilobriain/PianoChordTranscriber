def is_compatible_audio(filename):
    """
    Checks if the supplied filename has a WAV, MP3, or OGG file extension.
    Returns boolean result.
    """
    try:
        valid = ['wav', 'mp3', 'ogg']
        extension = filename[-3:].lower()
        if extension in valid:
            return True
        else:
            print("Invalid filetype")
            return False
    
    except Exception as e:
        print(f"is_compatible_audio function failure - {e}")
        return False