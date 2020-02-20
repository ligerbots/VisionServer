$(() => {
  loadCameraOnConnect({
    host: "192.168.122.36",
    container: '#cam-frame',
    port: 1190,
    image_url: '/stream.mjpg',
    data_url: '/settings.json',
    attrs: {
      width: 0,
      height: 0
    }
  });
  attachRobotConnectionIndicator("#status");
  function createItem(rt,tar,startclosed){

    var tag=$(`<div>${tar}</div>`);
    var cont=$(`<div style="margin-left:20px;"></div>`);

    var wrap=$(`<div></div>`)
    tag.click(()=>{
      cont.slideToggle();
    })
    wrap.append(tag);
    wrap.append(cont);
    rt.append(wrap);
    if(startclosed)cont.hide();
    return(cont)
  }
  function getType(itm){
    const prim_type=typeof itm;
    if(prim_type=="string"||prim_type=="boolean"||prim_type=="number"){
      return(prim_type);
    }else if(Array.isArray(itm)){
      const arr_type=typeof itm[0];
      if(arr_type=="string"||arr_type=="boolean"||arr_type=="number"){
        return(arr_type+" array")
      }
    }
    throw new TypeError(itm+" is not a valid type for networktables");
  }

  function toType(str,type){//returns null if invalid
    function toTypeSingle(str,type){
      switch (type) {
        case "string":
          return(str);
          break;
        case "boolean":
          const processed=str.trim().toLowerCase();
          if(processed=="true")return(true);
          else if(processed=="false")return(false);
          else return(null);
          break;
        case "number":
          const parsed=parseFloat(str);
          if(Number.isNaN(parsed))return(null);
          return(parsed)
          break;
        default:
          return(null);
      }
    }
    if(type.endsWith("array")){
      const arr_type=type.slice(0, -6);
      const splt=str.split(",");
      var res=[];
      for (var i = 0; i < splt.length; i++) {
        const proc=toTypeSingle(splt[i],arr_type);
        if(proc==null)return(null);
        res.push(proc);
      }
      return(res)
    }else{
      const prim_type=type;
      return(toTypeSingle(str,type));
    }
  }
  function additem(path,item,startclosed,fullpath,root){
    if(!path.length){
      const inp=$(`<input>`);
      inp.val(item);
      const type=getType(item);
      inp.change(()=>{
        const proc=toType(inp.val(),type);
        if(proc===null){
          inp.val(NetworkTables.getValue(fullpath));
        }else{
          NetworkTables.putValue(fullpath,proc);
        }
      });

      NetworkTables.addKeyListener(fullpath,function(key, value, isNew){7
        if(isNew)console.log("Unexpected: new value when not new");
        console.log("got change",key," expeced ",fullpath)

        inp.val(value);
      })
      root.append(inp);
      root.append(` (${type})`)
      return;
    }
    const rt=root||$("#nt-list");

    const rest=path.slice(1);
    const tar=path[0]
    let found=false;
    rt.children().each((indx,val)=>{
      let jval=$(val);
      if(jval.children().first().text()==tar){
        window.setTimeout(()=>{additem(rest,item,true,fullpath,jval.children().eq(1))},10);
        found=true;
      }
    })
    if(found)return found;
    additem(rest,item,true,fullpath,createItem(rt,tar,startclosed));
  }
  function splitpath(str){
    return str.split("/");
  }
  NetworkTables.addRobotConnectionListener(() => {
    window.setTimeout(() => {
      $("#nt-list").empty();
      let keys = NetworkTables.getKeys();
      keys.forEach((item) => {
        additem(splitpath(item),NetworkTables.getValue(item),false,item);
      });
    }, 100);
  });

})
