$(document).ready(function(){
    d3.json('data/{}', function(error, json) {
      if (error) throw error;
      render(json);
    });
});

function render(data){

    var inst = {};
    for(var k in data.institutes){
        var ins = data.institutes[k];
        inst[ins] = {'train': {}, 'test': {}}
        for(var i in data.folds){
            var fold = data.folds[i];
            for(var j in data.types){
                var type = data.types[j];
                inst[ins][type][fold] = 0;
                for(var r in data.data){
                    var row = data.data[r];
                    if(row.fold == fold && row.type == type && row.institute == ins){
                        inst[ins][type][fold]++;
                    }
                }
            }
        }
    }
    renderType(inst, data.folds, 'train')
    renderType(inst, data.folds, 'test')

}

function renderType(institutes, folds, type){

    var series = [];
    for(var i in institutes){
        var d = [];
        for(var f in folds){
            d.push(institutes[i][type][folds[f]]);
        }
        console.log(d);
        series.push({ name: i, data: d });
    }

    var fold_labels = [];
    for(var f in folds){
        fold_labels.push(folds[f] + '');
    }

    $('#' + type).highcharts({
        chart: {
            type: 'bar'
        },
        title: { text: type },
        xAxis: {
            categories: fold_labels
        },
        yAxis: {
            min: 0,
        },
        plotOptions: {
            series: {
                stacking: 'normal'
            }
        },
        series: series
    });
}