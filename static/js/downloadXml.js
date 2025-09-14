/**
 * Builder for MusicXML file that converts the chord names in the interface to a readable
 * XML file that can be opened in MuseScore, Sibelius etc.
 */

// Declare MusicXML output. Values added in resetBuilder()
let outputFile;
let measureCount;

/**
* Main function for building the MusicXML file
* from the chords table. The result is saved.
*/
function buildXml() {
  try {
    resetBuilder();

    let chords = document.getElementById("chords_table").innerText;
    let chordsArray = chords.split(/\t|\n/); // Split cells into chords

    // Iterate through chords and add to output
    chordsArray.forEach((chord) => {

      chordTriad = getTriadNotes(chord);
      alterTriad = getTriadAlters(chordTriad);

      addMeasure(chordTriad, alterTriad);
    });

    finishOutput();
    saveFile();

  } catch (error) {
    console.log(error);
    window.alert("Could not export to MusicXML. Check that chords are formatted correctly and try again.");
  }
}

/**
* Adds a chord triad to the current measure and begins a new one.
* step = root note (eg 'G') alter = 0 if natural, -1 if flat, 1 if sharp.
*/
function addMeasure(chordTriad, alterTriad) {
  // Increment measure count for next measure
  measureCount++;

  // Go through triad and add each note in measure
  for (let i = 0; i < chordTriad.length; i++) {
    let step = chordTriad[i].charAt(0); // get root note
    let alter = alterTriad[i]; // get alter info for flat, natural, or sharp

    outputFile += `
    <note>
	<chord/>
        <pitch>
          <step>${step}</step>
	  <alter>${alter}</alter>
          <octave>4</octave>
        </pitch>
        <duration>4</duration>
        <type>whole</type>
      </note>`
  }

  // Finish measure and begin new one      
  outputFile += `
    </measure>
    <measure number="${measureCount}">
    `
}

/**
* Finishes the output with a measure of rest as addMeasure begins a new
* measure automatically. Closes other necessary tags.
*/
function finishOutput() {
  outputFile += `
    <note>
        <rest/>
        <duration>4</duration>
        <type>whole</type>
    </note>
    </measure>
    </part>
    </score-partwise>`
}

/** 
* Saves the output in MusicXML format.
*/
function saveFile() {
  const text = outputFile;
  const blob = new Blob([text], { type: "application/vnd.recordare.musicxml+xml" });
  const link = document.createElement("a");

  link.href = URL.createObjectURL(blob);
  link.download = "chords.xml";
  link.click();
}

/**
* Resets the XML output
*/
function resetBuilder() {
  outputFile = `<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE score-partwise PUBLIC
    "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
    "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1">
      <part-name>Music</part-name>
    </score-part>
  </part-list>
  <part id="P1">
      <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key>
          <fifths>0</fifths>
        </key>
        <time>
          <beats>4</beats>
          <beat-type>4</beat-type>
        </time>
        <clef>
          <sign>G</sign>
          <line>2</line>
        </clef>
      </attributes>

      <direction placement="above">
        <direction-type>
          <metronome default-y="20" font-family="EngraverTextT" font-size="12" halign="left" relative-x="-32">
            <beat-unit>eighth</beat-unit>
            <per-minute>${bpm}</per-minute>
          </metronome>
        </direction-type>
        <sound tempo="${bpm}"/>
      </direction>
`;

  measureCount = 1;
}

/**
* Takes a chord and returns a triad of constituent notes
*/
function getTriadNotes(chord) {
  // Define semitone mapping
  const semitoneMap = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
  };

  const notes = [
    "C", "Db", "D", "Eb", "E", "F",
    "Gb", "G", "Ab", "A", "Bb", "B"
  ];

  // Extract root + quality
  const match = chord.match(/^([A-G][b#]?)(maj|min|sus2|sus4|dim|aug)$/i);
  if (!match) {
    console.warn(`Unknown chord: ${chord}`);
    return [];
  }

  const root = match[1];
  const quality = match[2].toLowerCase();

  // Intervals for triads (in semitones)
  const intervals = {
    "maj": [0, 4, 7],
    "min": [0, 3, 7],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8]
  };

  const rootSemitone = semitoneMap[root];
  const triad = intervals[quality].map(semi => notes[(rootSemitone + semi) % 12]);

  return triad;
}

/**
* Return an array of 'alter' values for a chord triad. This tag is
* used in MusicXML formatting to account for sharps and flats.
*/
function getTriadAlters(triad) {
  const alters = triad.map((note) => {
    if (note.charAt(1) === "b") {
      return "-1";
    } else if (note.charAt(1) === "#") {
      return "1";
    } else {
      return "0";
    }
  });
  return alters;
}