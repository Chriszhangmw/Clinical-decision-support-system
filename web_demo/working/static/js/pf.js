console.log($)

//console.log($('form').serializeArray())


let username = $('#form-first-name'); //病人姓名
let hospitalno = $('#form-last-number');//住院号
let describemain = $('#form-last-zhusu');//主诉
let describenow = $('#form-about-xianbingshi');//现病史
let describepast = $('#form-about-jiwangshi');//既往史
let descion_r = $('#form-about-yourself') //医生诊断
let cut = $('#cut') ;//分词结果
let predict = $('#predict')  ;//预测结果



let container_box = $('#container_box')  ;//盒子

$("#firstBtn").on("click", () => {
  let usernamen_txt = username.val();
  let hospitalnon_txt = hospitalno.val();
  let describemain_txt = describemain.val();
  let describenow_txt = describenow.val();
  let describepast_txt = describepast.val();
  if (usernamen_txt && hospitalnon_txt && describemain_txt && describenow_txt && describepast_txt) {
    let result = `${describemain_txt},${describenow_txt},${describepast_txt}`

    let pop=`<div id="pop" class="pop"><img src="static/img/load.png" alt=""></div>`
    container_box.append(pop)
    jQuery.post('./first', {data: result}, (data) => {
    /*  console.log(typeof(data))*/
      $('#pop').remove()

       data=JSON.parse(data)
      let predict_get = data.predict;
      let cut_get = data.cut;

/*console.log(typeof(cut_get))
console.log(cut_get)*/
      let predict_result = ``
      for (let key in predict_get) {
        predict_result += `<div> the ${key}: probability is  ${predict_get[key]}</div> `
      }
      predict.append(predict_result)

      let cut_result = ``;
      cut_get.forEach((str) => {
        cut_result += str + ' , ';
      })
      cut.text(cut_result)

    })
  } else {
    alert('请输入值')
  }

});


$("#submit_all").on("click", () => {
  let usernamen_txt = username.val();
  let hospitalnon_txt = hospitalno.val();
  let describemain_txt = describemain.val();
  let describenow_txt = describenow.val();
  let describepast_txt = describepast.val();
  let descion_txt = descion_r.val();
  if (usernamen_txt && hospitalnon_txt && describemain_txt && describenow_txt && describepast_txt) {
    let result = `${usernamen_txt},${hospitalnon_txt},${describemain_txt},${describenow_txt},${describepast_txt},${descion_txt}`
    jQuery.post('./secondAll', {data: result}, (data) => {
      console.log(data)
      username.val('')
      hospitalno.val('')
      describemain.val('')
      describenow.val('')
      describepast.val('')
      descion_r.val('')
      cut.html('')
      predict.html('')
    })

  } else {
    alert('请输入值')
  }

});