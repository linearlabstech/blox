let _emotions = ["happiness", "anger", "fun", "neutral", "hate", "worry", "love", "relief", "sadness", "boredom", "surprise", "enthusiasm", "empty"];
let _genders = ['female','male','unknown'];
let _ages = ['18-24','25-34','35-49','50-64','65-xx'];
let _sentiment = ['pos','neg'];
let _topics =  ["Relationships", "Food and Cooking", "Health", "Shopping", "Science", "Law and Legal Issues", "Cars and Vehicles", "Humor and Amusement", "History, Politics and Society", "Entertainment and Arts", "Hobbies and Collectibles", "Business and Finance", "Sports", "Travel and Places", "Technology", "Miscellaneous", "Jobs and Education", "Home and Garden", "Religion and Spirituality", "Literature and Language", "Animal Life"];
let table = {emotion:_emotions,gender:_genders,age:_ages,sentiment:_sentiments,topic:_topics}
function pie(data) {
    let obj = {emotions:[],genders:[],ages:[],sentiments:[],topics:[]};
    let pies = {emotion:{},gender:{},age:{},sentiment:{},topic:{}};
    for (pie in pies){
        for (key in table[pie]){
            pies[pie][key] = 0;
        }
    }
    for (d in data){
        for (key in d){
            pies[key][key]++;
        }
    }
    return pies;
}

function pie2data(obj){
    let array = [];
    for (key in obj) {
        array.push({
            value:obj[key],
            name:key
        });
    }
    return array;
}

function chartPie(obj,elms){
    for (elm in elms){
        var chart = echarts.init(document.getElementById(elm));

        var option = {
            tooltip: {},
            title:{text:elm},
            series: [{
                type: 'pie',
                roseType:'angle',
                data: pie2data(obj[elm]),
            }]
        };

        // use configuration item and data specified to show chart
        chart.setOption(option);
    }       
}