/**
 * Script to highlight relevant chord box as audio is played
 * in the chord interface.
 */

const secondsPerBeat = 60 / bpm;
const beatsPerBar = 4;
const secondsPerBar = secondsPerBeat * beatsPerBar;

const audio = document.getElementById("song_audio");

// keep track of which cell was last highlighted
let previousCell = null;

audio.addEventListener("timeupdate", () => {

    // figure out which bar (and thus cell) we’re in
    const barIndex = Math.floor(audio.currentTime / secondsPerBar);

    // un-highlight the old cell
    if (previousCell) {
        previousCell.classList.remove("highlight");
    }

    // highlight the new one
    const currentCell = document.getElementById(`cell${barIndex}`);
    if (currentCell) {
        currentCell.classList.add("highlight");
        previousCell = currentCell;
    }
});