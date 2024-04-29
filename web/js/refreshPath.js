import { app } from "../../../scripts/app.js";
import { api } from '../../../scripts/api.js'
import { ComfyWidgets } from "../../../scripts/widgets.js"
function rebootAPI() {
	if (confirm("Are you sure you'd like to reboot the server to refresh weights path?")) {
		try {
			api.fetchApi("/rvc/reboot");
		}
		catch(exception) {

		}
		return true;
	}

	return false;
}
function pathRefresh(node, inputName, inputData, app) {
    // Create the button widget for selecting the files
    let refreshWidget = node.addWidget("button", "REBOOT TO REFRESH SID LIST", "refresh", () => {
        rebootAPI()
    });

    refreshWidget.serialize = false;

    return { widget: refreshWidget };
}
ComfyWidgets.PATHREFRESH = pathRefresh;

app.registerExtension({
	name: "RVC.RefreshPath",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData?.name == "RVC_Infer") {
			nodeData.input.required.upload = ["PATHREFRESH"];
		}
	},
});