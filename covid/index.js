    const Server=require('node-http-server').Server;
 
    class MyCustomServer extends Server{
      constructor(){
        super();
      }
    }
 
    const server=new MyCustomServer;
    server.deploy();
 
