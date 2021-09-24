new Vue({
    el: '#app',
    data: {
        datasetName: "",
        webService: "",
        started: false,
        sessionName: '',
        topicDefinitions: [],
        iteration: 0,
        inputElement: null,
        outputElements: [],
        reward: null,
        previousOutputElementsMarkers: [],
        recommendations: [],
        nextElementType: "review",
        nextInputElement: null,
        nextRelevanceFunction: null,
        nextQualityFunction: null,
        terminateSession: false
    },
    mounted() {
        this.webService = window.location.origin
        let query_params = new URLSearchParams(window.location.search.substring(1));
        this.sessionName = query_params.get("session_name")
        this.datasetName = query_params.get("dataset_name")
        let url = new URL("start-exploration", this.webService)
        url.searchParams.append("dataset_name", this.datasetName)
        url.searchParams.append("session_name", this.sessionName)
        url.searchParams.append("keywords", query_params.get("keywords"))
        axios.get(url).then(response => {
            this.iteration = response.data.iteration
            this.inputElement = response.data.input_element
            this.outputElements = response.data.output_elements
            this.nextInputElement = response.data.output_elements[0]
            this.reward = response.data.reward
            this.previousOutputElementsMarkers = response.data.previous_output_elements_makers
            this.recommendations = response.data.recommendations
            this.nextRelevanceFunction = response.data.recommendations[1]
            this.nextQualityFunction = response.data.recommendations[0]
            this.topicDefinitions = response.data.topic_definitions
            this.started = true
        })
    },
    computed: {

    },
    methods: {
        executeNextStep() {
            let url = new URL("explore", this.webService)
            url.searchParams.append("dataset_name", this.datasetName)
            url.searchParams.append("session_name", this.sessionName)
            url.searchParams.append("input_element_id", this.nextInputElement.id)
            url.searchParams.append("relevance_function", this.nextRelevanceFunction)
            url.searchParams.append("quality_function", this.nextQualityFunction)
            url.searchParams.append("element_type", this.nextElementType)
            url.searchParams.append("iteration", this.iteration)
            url.searchParams.append("terminate_session", this.terminateSession)

            axios.get(url).then(response => {
                this.iteration = response.data.iteration
                this.inputElement = response.data.input_element
                this.outputElements = response.data.output_elements
                this.nextInputElement = response.data.output_elements[0]
                this.reward = response.data.reward
                this.previousOutputElementsMarkers = response.data.previous_output_elements_makers
                this.recommendations = response.data.recommendations
                this.nextRelevanceFunction = response.data.recommendations[1]
                this.nextQualityFunction = response.data.recommendations[0]
                setTimeout(scrollTo(0, 0), 0)
            })
        },
        formattedSentiments(original_sentiments) {
            const sentiments = []
            for (let i = 0; i < 10; i = i + 2) {
                sentiments.push(original_sentiments[i] + original_sentiments[i + 1])
            }
            const sentimentDistribution = this.softmax(sentiments)
            return sentimentDistribution.map((x) => Math.round(x * 100 * 100) / 100)
        },
        formattedTopics(original_topics) {
            const topicDistribution = this.softmax(original_topics)
            return topicDistribution.map((x) => Math.round(x * 100) / 100)
        },
        softmax(vector) {
            const exp = vector.map(Math.exp)
            const sum = _.sum(exp)
            return exp.map((x) => x / sum)
        },
        elementIconClasses(element) {
            if (element.type == 'item') {
                return ['fas', 'fa-6x', 'fa-cube']
            } else {
                return ['far', 'fa-6x', 'fa-comment']
            }
        }
    }
})