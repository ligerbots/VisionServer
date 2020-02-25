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
  $("#vrt-layout-btn").click(()=>{
    $("#content-area").removeClass("vert-split hor-split cam-fullscreen");
    $("#content-area").addClass("vert-split");
  });
  $("#hor-layout-btn").click(()=>{
    $("#content-area").removeClass("vert-split hor-split cam-fullscreen");
    $("#content-area").addClass("hor-split");
  });
  $("#fs-layout-btn").click(()=>{
    $("#content-area").removeClass("vert-split hor-split cam-fullscreen");
    $("#content-area").addClass("cam-fullscreen");
  });
  function createItem(rt, tar, startclosed, isleaf) {
    var tag = $(`<div>${tar}</div>`);
    var cont = $(`<div></div>`);

    var wrap = $(`<div></div>`)
    if (isleaf) {
      tag.addClass("leaf-tag");
      cont.addClass("leaf-itm");
      tag.addClass("optn-open")
    } else {
      tag.attr("tabindex","0");
      tag.addClass("stem-tag")
      cont.addClass("stem-itm")

      const clickhandler=(() => {
        if (cont.is(":hidden")) {
          tag.addClass("optn-open")
        } else {
          tag.removeClass("optn-open")
        }
        cont.slideToggle();
      });
      tag.click(clickhandler);
      tag.keypress(function (e) {
        if (e.key === ' ' || e.key === 'Spacebar'||event.key === "Enter") {
          clickhandler();
        }
      });

      if (startclosed) {
        cont.hide();
      } else {
        tag.addClass("optn-open")
      }
    }
    wrap.append(tag);
    wrap.append(cont);
    rt.append(wrap);

    return (cont)
  }

  function getType(itm) {
    const prim_type = typeof itm;
    if (prim_type == "string" || prim_type == "boolean" || prim_type == "number") {
      return (prim_type);
    } else if (Array.isArray(itm)) {
      if (!itm.length) return ("unknown array")
      const arr_type = typeof itm[0];
      if (arr_type == "string" || arr_type == "boolean" || arr_type == "number") {
        return (arr_type + " array")
      }
    }
    console.log(itm)
    throw new TypeError(itm + " (" + prim_type + ") is not a valid type for networktables");
  }

  function toType(str, type) { //returns null if invalid
    function toTypeSingle(str, type) {
      switch (type) {
        case "string":
          return (str);
          break;
        case "boolean":
          const processed = str.trim().toLowerCase();
          if (processed == "true") return (true);
          else if (processed == "false") return (false);
          else return (null);
          break;
        case "number":
          const parsed = parseFloat(str);
          if (Number.isNaN(parsed)) return (null);
          return (parsed)
          break;
        default:
          return (null);
      }
    }
    if (type.endsWith("array")) {
      const arr_type = type.slice(0, -6);
      const splt = str.split(",");
      var res = [];
      for (var i = 0; i < splt.length; i++) {
        const proc = toTypeSingle(splt[i], arr_type);
        if (proc == null) return (null);
        res.push(proc);
      }
      return (res)
    } else {
      const prim_type = type;
      return (toTypeSingle(str, type));
    }
  }

  function putKey(path, item, startclosed, fullpath, root) {
    if (!path.length) {
      if (root.children().length) {
        root.children().first().val(item);
        return;
      }
      const inp = $(`<input>`);
      inp.val(item);
      const type = getType(item);
      inp.change(() => {
        const proc = toType(inp.val(), type);
        if (proc === null) {
          inp.val(NetworkTables.getValue(fullpath));
        } else {
          NetworkTables.putValue(fullpath, proc);
        }
      });

      NetworkTables.addKeyListener(fullpath, function(key, value, isNew) {
        if (isNew) console.log("Unexpected: new value when not new");
        inp.val(value);
      })
      root.append(inp);
      root.append($(`<div class=leaf-itm-info>(${type})</div>`))
      return;
    }
    const rt = root || $("#nt-list");

    const rest = path.slice(1);
    const tar = path[0]
    let found = false;
    rt.children().each((indx, val) => {
      let jval = $(val);
      if (jval.children().first().text() == tar) {
        window.setTimeout(() => {
          putKey(rest, item, true, fullpath, jval.children().eq(1))
        }, 10);
        found = true;
      }
    })
    if (found) return found;
    putKey(rest, item, true, fullpath, createItem(rt, tar, startclosed, !rest.length));
  }

  function splitpath(str) {
    const split = str.split("/");
    return split[0] ? split : split.slice(1);
  }
  NetworkTables.addRobotConnectionListener(() => {
    window.setTimeout(() => {
      $("#nt-list").empty();
      const keys = NetworkTables.getKeys();
      keys.forEach((key) => {
        putKey(splitpath(key), NetworkTables.getValue(key), false, key);
      });
      {
        const prePath = "/SmartDashboard/vision/active_mode/";
        const camOpts = NetworkTables.getValue(prePath + "options");
        if (camOpts) {
          for (var i = 0; i < camOpts.length; i++) {
            const thisOpt = camOpts[i];
            const radioId = "camOptId" + thisOpt;
            let opt = $(`<input type="radio" id="${radioId}" name="camOpts">`);
            let lab = $(`<label for="${radioId}">${thisOpt}</label>`)
            $("#cam-selection").append(opt);
            $("#cam-selection").append(lab);
            $("#cam-selection").append($("<br>"));
            opt.change(() => {
              if (opt.is(':checked')) {
                NetworkTables.putValue(prePath + "selected", thisOpt);
              }
            });
          }
          const currSel = $("#camOptId" + NetworkTables.getValue(prePath + "selected"));
          currSel.prop('checked', true);
          NetworkTables.addKeyListener(prePath + "selected", (key, value) => {
            $(`#camOptId${value}`).prop('checked', true);
          });
        } else {
          $("#cam-selection").append("Unable to get active_mode options (check path?)")
        }
      }
    }, 100);
  });
  NetworkTables.addGlobalListener((key, value, isNew) => {
    if (!isNew) return;
    putKey(splitpath(key), NetworkTables.getValue(key), false, value);

  })
})
