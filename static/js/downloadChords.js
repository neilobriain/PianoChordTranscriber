/**
 * Download chords as a plain text file.
 */
function downloadChords() {
    const text = document.getElementById("chords_table").innerText;
    const blob = new Blob([text], { type: "text/plain" });
    const link = document.createElement("a");

    link.href = URL.createObjectURL(blob);
    link.download = "chords.txt";
    link.click();
}