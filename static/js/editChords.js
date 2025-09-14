/**
 * Script to allow editing of chord predictions in the interface.
 */

var editable = {
  // Flags
  ccell : null, // current editing cell
  cval : null, // current cell value

  // Edit cell
  edit : cell => {
    // Save selected cell
    editable.ccell = cell;
    editable.cval = cell.innerHTML;

    // Set editable
    cell.classList.add("edit");
    cell.contentEditable = true;
    cell.focus();

    // Listen to end edit
    cell.onblur = editable.done;
    cell.onkeydown = e => {
      if (e.key=="Enter") { editable.done(); }
      if (e.key=="Escape") { editable.done(1); }
    };
  },

  // Exit edit cell
  done : discard => {
    // Remove listeners
    editable.ccell.onblur = "";
    editable.ccell.onkeydown = "";

    // Stop edit
    editable.ccell.classList.remove("edit");
    editable.ccell.contentEditable = false;

    // Discard changes if esc
    if (discard===1) { editable.ccell.innerHTML = editable.cval; }

    // Save change check
    if (editable.ccell.innerHTML != editable.cval) {
      /* VALUE CHANGED */

      // If the user leaves the chord box blank, replace the <br> with a
      // space to improve saved file layout in plain text
      if (editable.ccell.innerHTML === '<br>') {
        editable.ccell.innerHTML = '&nbsp;';
      }

      console.log(editable.ccell.innerHTML);
    }
  }
};

// Double click to edit cell
window.addEventListener("load", () => {
  for (let td of document.querySelectorAll(".editable td")) {
    td.addEventListener("dblclick", () => editable.edit(td));
  }
});