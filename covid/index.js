const server=require('node-http-server');
 
server.deploy(
    {
        port:8000,
        root:'~/myApp/'
    },
    serverReady
);
 
server.deploy(
    {
        port:8888,
        root:'~/myOtherApp/'
    },
    serverReady
);
 
function serverReady(server){
   console.log( `Server on port ${server.config.port} is now up`);
}
