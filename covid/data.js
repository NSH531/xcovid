const spawn = require("child_process").spawn;
const pythonProcess = spawn('python',["test.py"]);
alert(pythonProcess);
