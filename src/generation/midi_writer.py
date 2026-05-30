from pathlib import Path

from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo

from src.preprocessing.vocab import BAR_TOKEN, END_TOKEN, START_TOKEN


def parse_music_token(token):
    parts = token.split("_")
    kind = parts[0]
    if kind == "REST" and len(parts) == 2:
        try:
            return kind, 0, int(parts[1])
        except ValueError:
            return None
    if kind == "NOTE" and len(parts) == 3:
        try:
            return kind, int(parts[1]), int(parts[2])
        except ValueError:
            return None
    if kind not in {"NOTE", "REST"}:
        return None
    return None


def tokens_to_events(tokens):
    events = []
    for token in tokens:
        if token in {START_TOKEN, END_TOKEN, BAR_TOKEN} or token.startswith("CHORD_"):
            continue
        parsed = parse_music_token(token)
        if parsed:
            events.append(parsed)
    return events


def tokens_to_midi(tokens, output_path, ticks_per_unit=120, velocity=72, bpm=120):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    midi = MidiFile(ticks_per_beat=ticks_per_unit * 4)
    track = MidiTrack()
    midi.tracks.append(track)
    track.append(MetaMessage("set_tempo", tempo=bpm2tempo(bpm), time=0))

    pending_time = 0
    for kind, pitch, duration_units in tokens_to_events(tokens):
        ticks = int(duration_units * ticks_per_unit)
        if kind == "REST":
            pending_time += ticks
            continue
        track.append(Message("note_on", note=pitch, velocity=velocity, time=pending_time))
        track.append(Message("note_off", note=pitch, velocity=0, time=ticks))
        pending_time = 0

    track.append(MetaMessage("end_of_track", time=pending_time))
    midi.save(output_path)
    return output_path
